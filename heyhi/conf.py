#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools to load a config with overrides and includes."""
from typing import Any, Dict, List, Sequence, Tuple, TypeVar, Union
import logging
import pathlib
import re

import google.protobuf.descriptor
import google.protobuf.json_format
import google.protobuf.message
import google.protobuf.text_format

import conf.conf_pb2
import conf.conf_cfgs

AnyPath = Union[pathlib.Path, str]
ProtoMessage = google.protobuf.message.Message

CONF_ROOT = pathlib.Path(conf.conf_pb2.__file__).parent
PROJ_ROOT = pathlib.Path(__file__).parent.parent
EXT = ".prototxt"
INCLUDES_FIELD = "includes"
INCLUDE_KEY = "I"
UNSET_VALUE = "NULL"
SELECT_ONEOF_VALUE = "SELECT"
ROOT_CFG_CLASS = conf.conf_pb2.MetaCfg

PROTO_TYPE_TO_FLOAT = {
    google.protobuf.descriptor.FieldDescriptor.TYPE_DOUBLE: float,
    google.protobuf.descriptor.FieldDescriptor.TYPE_FLOAT: float,
    google.protobuf.descriptor.FieldDescriptor.TYPE_INT64: int,
    google.protobuf.descriptor.FieldDescriptor.TYPE_UINT64: int,
    google.protobuf.descriptor.FieldDescriptor.TYPE_INT32: int,
    google.protobuf.descriptor.FieldDescriptor.TYPE_BOOL: bool,
    google.protobuf.descriptor.FieldDescriptor.TYPE_STRING: str,
    google.protobuf.descriptor.FieldDescriptor.TYPE_UINT32: int,
    google.protobuf.descriptor.FieldDescriptor.TYPE_ENUM: int,
}


def overrides_to_dict(overrides: Sequence[str]) -> Dict[str, str]:
    d = {}
    for override in overrides:
        try:
            name, value = override.split("=", 1)
        except ValueError:
            raise ValueError(f"Bad override: {override}. Expected format: key=value")
        d[name] = value
    return d


def _resolve_include(
    path: str, include_dirs: Sequence[pathlib.Path], mount_point: str
) -> pathlib.Path:
    """Tries to find the config in include_dirs and returns full path.

    path is either a full path or a relative path (relive to one of include_dirs)
    """
    if path.startswith("/"):
        full_path = pathlib.Path(path)
        if not full_path.exists():
            raise ValueError(f"Cannot find include path {path}")
        return full_path
    if "/" in path and (PROJ_ROOT / path).exists():
        return PROJ_ROOT / path
    if path.endswith(EXT):
        path = path[: -len(EXT)]
    possible_includes = []
    mount_point = mount_point.strip(".")
    if mount_point:
        include_dirs = list(include_dirs) + [
            p / mount_point.replace(".", "/") for p in include_dirs
        ]
    for include_path in include_dirs:
        full_path = include_path / (path + EXT)
        if full_path.exists():
            return full_path
        elif full_path.parent.exists():
            possible_includes.extend(
                str(p.resolve())[len(str(include_path.resolve())) : -len(EXT)].lstrip("/")
                for p in full_path.parent.iterdir()
                if str(p).endswith(EXT)
            )

    err_msg = f"Cannot find include {path}"
    if possible_includes:
        err_msg += ". Possible typo, known includes:\n%s" % "\n".join(possible_includes)
    raise ValueError(err_msg)


def _parse_overrides(
    overrides: Sequence[str], include_dirs: Sequence[pathlib.Path]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Separate overrides into include-overrides and scalar overrides."""
    override_dict = overrides_to_dict(overrides)
    include_dict, scalar_dict = {}, {}
    for key, value in override_dict.items():
        if key.startswith(INCLUDE_KEY):
            key = key[1:].lstrip(".")
            value = str(_resolve_include(value, include_dirs, key))
            include_dict[key] = value
        else:
            scalar_dict[key] = value
    return include_dict, scalar_dict


def _get_sub_config(cfg: ProtoMessage, mount: str) -> ProtoMessage:
    if not mount:
        return cfg
    subcfg = cfg
    for key in mount.split("."):
        if not hasattr(subcfg, key):
            raise ValueError("Cannot resolve path '%s' in config:\n%s" % (mount, cfg))
        subcfg = getattr(subcfg, key)
    return subcfg


def _edit_distance(str1: str, str2: str) -> int:
    # Putting import inside to make CI test for config validity faster.
    import numpy as np

    distances = np.zeros((len(str1) + 1, len(str2) + 1), "int32")
    distances[0] = np.arange(len(distances[0]))
    distances[:, 0] = np.arange(len(distances[:, 0]))
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            distances[i, j] = min(
                distances[i - 1, j] + 1,
                distances[i, j - 1] + 1,
                distances[i - 1, j - 1] + int(str1[i - 1] != str2[j - 1]),
            )
    return distances[-1, -1]


def flatten_cfg(cfg: ProtoMessage, *, with_all=True) -> Dict[str, Any]:
    def _flatten(cfg_dict):
        if not isinstance(cfg_dict, dict):
            yield ("", cfg_dict)
        else:
            for k, v in cfg_dict.items():
                yield from ((f"{k}.{suffix}".strip("."), value) for suffix, value in _flatten(v))

    return dict(sorted(_flatten(cfg.to_dict(with_all=with_all))))


def _find_possible_typos(cfg: ProtoMessage, mount: str) -> List[str]:
    def dist(known_mount):
        return _edit_distance(mount, known_mount)

    return sorted(flatten_cfg(cfg), key=dist)


def _apply_scalar_override(cfg: ProtoMessage, mount: str, value: str) -> None:
    assert mount, "Scalar override with empty key!"
    # We want something like recursive_seattr(cfg, mount, value). But we
    # need to handle the recursive parts and also cast value to correct
    # type.
    mount_parent, key = mount.rsplit(".", 1) if "." in mount else ("", mount)
    subcfg = _get_sub_config(cfg, mount_parent)
    if type(subcfg).__name__ == "ScalarMapContainer":
        # Shortcut for maps.
        subcfg[key] = value
        return
    if type(subcfg).__name__ == "RepeatedScalarContainer":
        # Shortcut for arrays.
        try:
            key = int(key)
        except ValueError:
            raise ValueError(f"Got non-integer key {key} for repeated feild {mount_parent}")
        if key != -1 and not 0 <= key <= len(subcfg):
            raise ValueError(
                f"Cannot acess element {key} in list {mount_parent} that has {len(subcfg)}"
                " elements. Use '-1' to append an element"
            )
        if key == -1 or key == len(subcfg):
            subcfg.append(value)
        else:
            subcfg[key] = value
        return

    if not hasattr(subcfg, key):
        possible_typos = _find_possible_typos(cfg, mount)[:5]
        raise ValueError(
            "Cannot resolve path '%s' in config:\n%s\nPossible typos: %s"
            % (mount, cfg, possible_typos)
        )
    if value == UNSET_VALUE:
        subcfg.ClearField(key)
        return
    field = subcfg.DESCRIPTOR.fields_by_name[key]
    if field.message_type is not None:
        if value == SELECT_ONEOF_VALUE and field.containing_oneof is not None:
            # Merge empty message of the submessage's type into the submessage.
            getattr(subcfg, key).MergeFrom(type(getattr(subcfg, key))())
            return
        raise ValueError("Trying to set scalar '%s' for message type '%s'" % (value, mount))
    attr_type = PROTO_TYPE_TO_FLOAT[field.type]
    if attr_type is bool:
        value = value.lower()
        assert value in ("true", "false", "0", "1"), value
        value = True if value in ("true", "1") else False
    elif attr_type is int and not value.isdigit():
        # If enum is redefined we have to search in the parrent object
        # for all enums.
        if field.enum_type is not None:
            value = field.enum_type.values_by_name[value].number
    try:
        value = attr_type(value)
    except ValueError:
        raise ValueError(
            "Value for %s should be of type %s. Cannot cast provided value %s to this type"
            % (mount, attr_type, value)
        )
    setattr(subcfg, key, value)


def _parse_text_proto_into(path, msg):
    with path.open() as stream:
        proto_text = stream.read()
    proto_text = re.sub(r"\{\{ *ROOT_DIR *\}\}", str(PROJ_ROOT), proto_text)
    try:
        google.protobuf.text_format.Merge(proto_text, msg)
    except google.protobuf.text_format.ParseError:
        logging.error(
            "Got an exception while parsing proto from %s into type %s. Proto text:\n%s",
            path,
            type(msg),
            proto_text,
        )
        raise


def _guess_message_type(path):
    logging.info("Going to guess message type by trying all of them")
    with path.open() as stream:
        proto_text = stream.read()
    proto_text = re.sub(r"\{\{ *ROOT_DIR *\}\}", str(PROJ_ROOT), proto_text)
    msg_types_and_errors = []
    for msg_type in reversed(google.protobuf.message.Message.__subclasses__()):
        try:
            google.protobuf.text_format.Merge(proto_text, msg_type())
        except google.protobuf.text_format.ParseError as e:
            msg_types_and_errors.append((msg_type, e))
            continue
        logging.info("Guessed type: %s", msg_type)
        return msg_type
    msg_types_and_errors_formatted = "\n".join(
        f"{msg_type}: {e}" for msg_type, e in msg_types_and_errors
    )
    raise ValueError(
        f"Failed to guess message type for {path}. Please check the type you think it should be and see what the error is: {msg_types_and_errors_formatted}"
    )


def _apply_include(msg, mount, include_msg_path, include_dirs):
    sub_msg = _get_sub_config(msg, mount)
    include_msg_path = _resolve_include(include_msg_path, include_dirs, mount)
    logging.debug(
        "Constructing %s. Applying include: mount=%r include=%r subcfg=%s",
        type(msg).__name__,
        mount,
        include_msg_path,
        type(sub_msg).__name__,
    )
    sub_msg.MergeFrom(load_proto_message(include_msg_path, msg_class=type(sub_msg)))


def _get_task_type(config_path: pathlib.Path) -> str:
    root_cfg = ROOT_CFG_CLASS()
    _parse_text_proto_into(config_path, root_cfg)
    task = root_cfg.WhichOneof("task")
    if not task:
        raise ValueError("Bad config - no specific config specified:\n%s" % root_cfg)
    return task


def _get_config_includes(config_path: pathlib.Path, msg_class) -> Dict[str, str]:
    """Returns a dict (mount -> path) from the config."""
    msg = msg_class()
    _parse_text_proto_into(config_path, msg)
    return dict((x.mount, x.path) for x in getattr(msg, INCLUDES_FIELD, []))


def load_proto_message(
    config_path: AnyPath,
    overrides: Sequence[str] = tuple(),
    *,
    msg_class=None,
    extra_include_dirs: Sequence[pathlib.Path] = tuple(),
) -> ProtoMessage:
    """Loads message from the file and applies overrides.

    If message type is not give, will try to guess message type.

    All includes in the loaded messages (root message and includes) will be
    recursively included.

    Composition order:
      * Create empty message of type msg_class
      * Merge includes within the message in config_path
      * Merge the content of confif_path
      * Merge includes in overrides
      * Merge scalars in overrides

    Returns the message.
    """
    config_path = pathlib.Path(config_path)
    if msg_class is None:
        msg_class = _guess_message_type(config_path)
    elif hasattr(msg_class, "get_proto_class"):
        msg_class = msg_class.get_proto_class()

    def _resolve_mount(mount):
        if msg_class is ROOT_CFG_CLASS:
            # For convinience, top-level includes do not include name of task, i.e.,
            # `lr=XXX` vs `train_sl.lr=XXX` where train_sl is a name of task. We
            # manually add it.
            return (_get_task_type(config_path) + "." + mount).strip(".")
        else:
            return mount

    include_dirs = []
    include_dirs.append(config_path.resolve().parent)
    include_dirs.append(CONF_ROOT / "common")
    include_dirs.extend(extra_include_dirs)

    include_overides, scalar_overideds = _parse_overrides(overrides, include_dirs=include_dirs)
    logging.debug(
        "Constructing %s from %s with include overrides %s and scalar overrides %s",
        msg_class.__name__,
        config_path,
        include_overides,
        scalar_overideds,
    )

    msg = msg_class()

    # Step 1: Populate message with includes.
    default_includes = _get_config_includes(config_path, msg_class)
    logging.debug("%s defaults %s", msg_class, default_includes)
    for mount, include_msg_path in default_includes.items():
        _apply_include(msg, _resolve_mount(mount), include_msg_path, include_dirs)

    # Step 2: Override the includes with the config content.
    _parse_text_proto_into(config_path, msg)
    if hasattr(msg, INCLUDES_FIELD):
        msg.ClearField(INCLUDES_FIELD)

    # Step 3: Override with extra includes.
    for mount, include_msg_path in include_overides.items():
        _apply_include(msg, _resolve_mount(mount), include_msg_path, include_dirs)

    # Step 4: Apply scalar overrides.
    for mount, value in scalar_overideds.items():
        logging.debug(
            "Constructing %s. Applying scalar: mount=%r value=%r",
            msg_class.__name__,
            _resolve_mount(mount),
            value,
        )
        _apply_scalar_override(msg, _resolve_mount(mount), value)
    return msg


def load_config(
    config_path: AnyPath,
    overrides: Sequence[str] = tuple(),
    *,
    msg_class=None,
    extra_include_dirs: Sequence[pathlib.Path] = tuple(),
) -> "conf.conf_pb2._FrozenConf":
    """Loads a config from the path and applies overrides.

    If msg_class is None, will try to autodetect message type by trying to
        parse the file with all possible message types.
    If msg_class provided, it should be either a proto message class or a frozen config
        class. It's better to specity the class explicitly to avoid surprises.

    Returns a frozen config.
    """
    cfg = load_proto_message(
        config_path,
        overrides=overrides,
        msg_class=msg_class,
        extra_include_dirs=extra_include_dirs,
    )
    return cfg.to_frozen()


def load_root_config(
    config_path: AnyPath,
    overrides: Sequence[str] = tuple(),
    *,
    extra_include_dirs: Sequence[pathlib.Path] = tuple(),
) -> conf.conf_cfgs.MetaCfg:
    """A shortcut for load_config for MetaCfg."""
    return load_config(
        config_path, overrides, msg_class=ROOT_CFG_CLASS, extra_include_dirs=extra_include_dirs
    )


def load_root_proto_message(
    config_path: AnyPath,
    overrides: Sequence[str],
    extra_include_dirs: Sequence[pathlib.Path] = tuple(),
) -> Tuple[str, ROOT_CFG_CLASS]:
    config_path = pathlib.Path(config_path)
    task = _get_task_type(config_path)
    meta_cfg = load_proto_message(
        config_path,
        overrides=overrides,
        msg_class=ROOT_CFG_CLASS,
        extra_include_dirs=extra_include_dirs,
    )
    return task, meta_cfg


def save_config(cfg: ProtoMessage, path: pathlib.Path):
    if cfg.is_frozen():
        return save_config(cfg._msg, path)
    with path.open("w") as stream:
        stream.write(google.protobuf.text_format.MessageToString(cfg))
        stream.write("\n")


def conf_is_set(cfg: ProtoMessage, path: str) -> bool:
    """Returns true if value for path is set explicitly in the config."""
    assert not cfg.is_frozen()
    *components, name = path.split(".")
    subcfg = cfg
    for c in components:
        subcfg = getattr(subcfg, c)
    assert hasattr(subcfg, name), cfg
    assert not isinstance(getattr(subcfg, name), google.protobuf.message.Message), cfg
    return subcfg.HasField(name)


def conf_get(cfg: ProtoMessage, path: str) -> Any:
    """Returns value for the path.

    Note, using this on oneof fields may change the config.
    """
    assert not cfg.is_frozen()
    subcfg = cfg
    for c in path.split("."):
        subcfg = getattr(subcfg, c)
    return subcfg


def conf_set(cfg: ProtoMessage, path: str, value: Any) -> None:
    assert not cfg.is_frozen()
    _apply_scalar_override(cfg, path, str(value))


def conf_to_dict(cfg, include_defaults=False):
    if cfg.is_frozen():
        return conf_to_dict(cfg._msg, include_defaults=include_defaults)
    return google.protobuf.json_format.MessageToDict(
        cfg, preserving_proto_field_name=True, including_default_value_fields=include_defaults
    )


T = TypeVar("T", bound=ProtoMessage)


def conf_with_overrides(cfg: T, overrides: List[str]) -> T:
    assert cfg.is_frozen()  # Would edit in place if the config was editable!
    cfg = cfg.to_editable()
    for override in overrides:
        key, value = override.split("=", 1)
        assert not key.startswith("I"), f"Got include overrides that is not supported: {key}"
        conf_set(cfg, key, value)
    return cfg.to_frozen()
