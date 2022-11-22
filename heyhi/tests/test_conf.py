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

"""Tests for HeyHi configuration.

Run with pytest.
"""
import pathlib
import pickle
import unittest

import heyhi.conf
import conf.conf_pb2
import conf.conf_cfgs


class TestRootConf(unittest.TestCase):
    def _load(self, overrides):
        data_dir = pathlib.Path(__file__).parent / "data"
        root_cfg = data_dir / "root.prototxt"
        task, meta_cfg = heyhi.conf.load_root_proto_message(root_cfg, overrides)
        self.assertEqual(task, "test")
        return getattr(meta_cfg, task)

    def testLoadSimple(self):
        cfg = self._load(overrides=[])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, -1)

    def testScalarOverride(self):
        cfg = self._load(overrides=["scalar=20"])
        self.assertEqual(cfg.scalar, 20)
        self.assertEqual(cfg.sub.subscalar, -1)

    def testScalarOverrideEnum(self):
        cfg = self._load(overrides=["enum_value=ONE"])
        self.assertEqual(cfg.enum_value, 1)
        self.assertEqual(cfg.to_frozen().enum_value, "ONE")

    def testScalarOverrideFloat(self):
        cfg = self._load(overrides=["scalar=20.5"])
        self.assertEqual(cfg.scalar, 20.5)
        self.assertEqual(cfg.sub.subscalar, -1)

    def testScalarOverrideSub(self):
        cfg = self._load(overrides=["sub.subscalar=20"])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, 20)

    def testScalarOverrideTypeCast(self):
        self.assertRaises(ValueError, lambda: self._load(overrides=["sub.subscalar=20.5"]))

    def testIncludeScalar(self):
        cfg = self._load(overrides=["I=redefine_scalar"])
        self.assertEqual(cfg.scalar, 1.0)
        self.assertEqual(cfg.sub.subscalar, -1)

    def testIncludeMessage(self):
        cfg = self._load(overrides=["I=redefine_message"])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, 22)
        self.assertEqual(cfg.sub2.subscalar, -1)

    def testIncludeMessageWithMatchingPath(self):
        self._load(overrides=["Ilauncher=local"])

    def testIncludeMessageInside(self):
        cfg = self._load(overrides=["I.sub=redefine_subscalar_22"])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, 22)
        self.assertEqual(cfg.sub2.subscalar, -1)

    def testIncludeMessageInsideTwice(self):
        cfg = self._load(overrides=["I.sub=redefine_subscalar_22", "I.sub2=redefine_subscalar_22"])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, 22)
        self.assertEqual(cfg.sub2.subscalar, 22)


class TestRootWithIncludesConf(unittest.TestCase):
    def _load(self, overrides):
        data_dir = pathlib.Path(__file__).parent / "data"
        root_cfg = data_dir / "root_with_includes.prototxt"
        task, meta_cfg = heyhi.conf.load_root_proto_message(root_cfg, overrides)

        self.assertEqual(task, "test")
        return getattr(meta_cfg, task)

    def testLoadSimple(self):
        cfg = self._load(overrides=[])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, -1)
        self.assertEqual(cfg.sub2.subscalar, 22)

    def testLoadOverrideInclude(self):
        cfg = self._load(overrides=["I.sub2=redefine_subscalar_37"])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, -1)
        self.assertEqual(cfg.sub2.subscalar, 37)


class TestRootWithIncludesAndRedefinesConf(unittest.TestCase):
    def _load(self, overrides):
        data_dir = pathlib.Path(__file__).parent / "data"
        root_cfg = data_dir / "root_with_includes_redefined.prototxt"
        meta_cfg = heyhi.conf.load_proto_message(
            root_cfg, overrides=overrides, msg_class=conf.conf_pb2.MetaCfg
        )
        task = meta_cfg.WhichOneof("task")
        self.assertEqual(task, "test")
        return getattr(meta_cfg, task)

    def testLoadSimple(self):
        cfg = self._load(overrides=[])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, 22)
        self.assertEqual(cfg.sub2.subscalar, 99)


class TestRootWithIncludesClearField(unittest.TestCase):
    def _load(self, overrides):
        # scalar_no_default: 42.0
        # sub {
        #   subscalar: 9
        # }
        # oneof_value1: 11
        data_dir = pathlib.Path(__file__).parent / "data"
        root_cfg = data_dir / "root_densely_filled.prototxt"
        task, meta_cfg = heyhi.conf.load_root_proto_message(root_cfg, overrides)
        self.assertEqual(task, "test")
        return getattr(meta_cfg, task)

    def testUnsetScalar(self):
        cfg = self._load(overrides=["sub.subscalar=NULL"])
        print(cfg)
        self.assertFalse(cfg.sub.HasField("subscalar"))
        self.assertTrue(cfg.HasField("sub"))

    def testUnsetMessage(self):
        cfg = self._load(overrides=["sub=NULL"])
        print(cfg)
        self.assertFalse(cfg.sub.HasField("subscalar"))
        self.assertFalse(cfg.HasField("sub"))

    def testUnsetOneof(self):
        cfg = self._load(overrides=["oneof_value1=NULL"])
        self.assertFalse(cfg.HasField("oneof_value1"))
        self.assertFalse(cfg.HasField("oneof_field"))

    def testUnsetOneofNoop(self):
        # Unsetting non-chosed field value2 instead of value1.
        cfg = self._load(overrides=["oneof_value2=NULL"])
        self.assertTrue(cfg.HasField("oneof_value1"))
        self.assertTrue(cfg.HasField("oneof_field"))


class TestRootWithIncludesSelectField(unittest.TestCase):
    def _load(self, overrides):
        # scalar_no_default: 42.0
        # sub {
        #   subscalar: 9
        # }
        # oneof_value1: 11
        data_dir = pathlib.Path(__file__).parent / "data"
        root_cfg = data_dir / "root_densely_filled.prototxt"
        task, meta_cfg = heyhi.conf.load_root_proto_message(root_cfg, overrides)
        self.assertEqual(task, "test")
        return getattr(meta_cfg, task)

    def test(self):
        cfg = self._load(overrides=["oneof_value3=SELECT"])
        print(cfg)
        self.assertFalse(cfg.HasField("oneof_value1"))
        # oneof_value3 is set, but no inner fields are initialized.
        self.assertTrue(cfg.HasField("oneof_value3"))
        self.assertFalse(cfg.oneof_value3.HasField("subscalar"))


class TestPathGetters(unittest.TestCase):
    def _load(self, overrides):
        data_dir = pathlib.Path(__file__).parent / "data"
        root_cfg = data_dir / "root.prototxt"

        task, meta_cfg = heyhi.conf.load_root_proto_message(root_cfg, overrides)
        self.assertEqual(task, "test")
        return getattr(meta_cfg, task)

    def testConfIsSet(self):
        cfg = self._load(overrides=["sub.subscalar=1", "oneof_value1=1"])
        self.assertEqual(heyhi.conf.conf_is_set(cfg, "scalar"), False)
        self.assertEqual(heyhi.conf.conf_is_set(cfg, "sub.subscalar"), True)
        self.assertEqual(heyhi.conf.conf_is_set(cfg, "sub2.subscalar"), False)
        self.assertRaises(AssertionError, heyhi.conf.conf_is_set, cfg, "bad_scalar")

        # Duplicating queries to make sure we are not altering config.
        self.assertEqual(heyhi.conf.conf_is_set(cfg, "oneof_value1"), True)
        self.assertEqual(heyhi.conf.conf_is_set(cfg, "oneof_value2"), False)
        self.assertEqual(heyhi.conf.conf_is_set(cfg, "oneof_value3.subscalar"), False)
        self.assertEqual(heyhi.conf.conf_is_set(cfg, "oneof_value1"), True)
        self.assertEqual(heyhi.conf.conf_is_set(cfg, "oneof_value2"), False)
        self.assertEqual(heyhi.conf.conf_is_set(cfg, "oneof_value3.subscalar"), False)

    def testConfGet(self):
        cfg = self._load(overrides=["sub.subscalar=1", "oneof_value1=1"])
        self.assertEqual(heyhi.conf.conf_get(cfg, "scalar"), -1)
        self.assertEqual(heyhi.conf.conf_get(cfg, "sub.subscalar"), 1)
        self.assertEqual(heyhi.conf.conf_get(cfg, "sub2.subscalar"), -1)

    def testConfSet(self):
        cfg = self._load(overrides=["sub.subscalar=1", "oneof_value1=1"])
        heyhi.conf.conf_set(cfg, "scalar", 10)
        self.assertEqual(cfg.scalar, 10)
        heyhi.conf.conf_set(cfg, "sub.subscalar", 11)
        self.assertEqual(cfg.sub.subscalar, 11)
        heyhi.conf.conf_set(cfg, "sub2.subscalar", 12)
        self.assertEqual(cfg.sub2.subscalar, 12)


class TestInnerRootWithIncludesConf(unittest.TestCase):
    def _load(self, overrides):
        data_dir = pathlib.Path(__file__).parent / "data"
        root_cfg = data_dir / "root_with_includes.prototxt"
        task, meta_cfg = heyhi.conf.load_root_proto_message(root_cfg, overrides)
        self.assertEqual(task, "test")
        return getattr(meta_cfg, task)

    def testLoadOverrideInclude(self):
        cfg = self._load(overrides=["I.complex_sub=redefine_message_with_include"])
        print(cfg)
        self.assertEqual(len(cfg.complex_sub.includes), 0)
        self.assertEqual(cfg.complex_sub.sub.subscalar, 37)


class TestLoadSingleMessage(unittest.TestCase):
    def setUp(self):
        self.data_dir = pathlib.Path(__file__).parent / "data"

    def testLoadPartialMessage(self):
        # Check that during message loading we resolve includes.
        cfg = heyhi.conf.load_proto_message(
            self.data_dir / "redefine_message_with_include.prototxt",
            msg_class=conf.conf_pb2.TestTask.ComplexSubmessageWithIncludes,
        )
        self.assertEqual(len(cfg.includes), 0)
        self.assertEqual(cfg.sub.subscalar, 37)

    def testLoadFullTaskMessage(self):
        # Check that during message loading we resolve includes.
        cfg = heyhi.conf.load_proto_message(
            self.data_dir / "root.prototxt", overrides=["scalar=10"]
        )
        self.assertEqual(cfg.test.scalar, 10)

    def testLoadAsFrozen(self):
        cfg = heyhi.conf.load_config(self.data_dir / "root.prototxt", overrides=["scalar=10"])
        self.assertTrue(cfg.is_frozen())
        self.assertEqual(cfg.test.scalar, 10)

        # Load wiht proto msg_class.
        cfg = heyhi.conf.load_config(
            self.data_dir / "root.prototxt",
            overrides=["scalar=10"],
            msg_class=conf.conf_pb2.MetaCfg,
        )
        # Load with frozen msg_class.
        cfg = heyhi.conf.load_config(
            self.data_dir / "root.prototxt",
            overrides=["scalar=10"],
            msg_class=conf.conf_cfgs.MetaCfg,
        )

    def testLoadFullTaskWithIncludesMessage(self):
        # Check that during message loading we resolve includes.
        cfg = heyhi.conf.load_proto_message(
            self.data_dir / "root_with_includes_redefined.prototxt"
        )
        self.assertEqual(cfg.test.sub.subscalar, 22)
        self.assertEqual(cfg.test.sub2.subscalar, 99)


class TestPatchedFunctions(unittest.TestCase):
    def setUp(self):
        self.data_dir = pathlib.Path(__file__).parent / "data"

    def testToDict(self):
        root_cfg = self.data_dir / "root.prototxt"
        cfg = heyhi.conf.load_proto_message(root_cfg)
        print(cfg)
        # test {
        #    enum_value: ZERO
        # }
        self.assertEqual(cfg.to_dict(), {"test": {"enum_value": "ZERO"}})
        self.assertEqual(
            cfg.to_dict(with_defaults=True),
            {
                "test": {
                    "complex_sub": {"sub": {"subscalar": -1}},
                    "enum_value": "ZERO",
                    "scalar": -1.0,
                    "sub": {"subscalar": -1},
                    "sub2": {"subscalar": -1},
                    "bool_field_with_default_false": False,
                    "bool_field_with_default_true": True,
                }
            },
        )
        self.assertEqual(
            cfg.to_dict(with_all=True),
            {
                "includes": [],
                "test": {
                    "complex_sub": {"includes": [], "sub": {"subscalar": -1}},
                    "enum_value": "ZERO",
                    "enum_value_no_default": None,
                    "scalar": -1.0,
                    "scalar_no_default": None,
                    "some_map": {},
                    "sub": {"subscalar": -1},
                    "sub2": {"subscalar": -1},
                    "bool_field_no_default": False,
                    "bool_field_with_default_false": False,
                    "bool_field_with_default_true": True,
                    "launcher": {},
                },
            },
        )

    def testToDictUnsetEnum(self):
        root_cfg = self.data_dir / "root.prototxt"
        cfg = heyhi.conf.load_proto_message(root_cfg)
        cfg.test.ClearField("enum_value")
        print(cfg)
        # test {
        # }
        self.assertEqual(cfg.to_dict(), {"test": {}})
        self.assertEqual(
            cfg.to_dict(with_all=True),
            {
                "includes": [],
                "test": {
                    "complex_sub": {"includes": [], "sub": {"subscalar": -1}},
                    "enum_value": "ZERO",
                    "enum_value_no_default": None,
                    "scalar": -1.0,
                    "scalar_no_default": None,
                    "some_map": {},
                    "sub": {"subscalar": -1},
                    "sub2": {"subscalar": -1},
                    "bool_field_no_default": False,
                    "bool_field_with_default_false": False,
                    "bool_field_with_default_true": True,
                    "launcher": {},
                },
            },
        )

    def testToDictMap(self):
        root_cfg = self.data_dir / "root.prototxt"
        cfg = heyhi.conf.load_proto_message(root_cfg)
        cfg.test.some_map["field1"] = 1
        cfg.test.some_map["field2"] = 2
        print(cfg)
        # test {
        # enum_value: ZERO
        # some_map {
        #     key: "field1"
        #     value: 1
        # }
        # some_map {
        #     key: "field2"
        #     value: 2
        # }

        self.assertEqual(
            cfg.to_dict(), {"test": {"enum_value": "ZERO", "some_map": {"field1": 1, "field2": 2}}}
        )
        self.assertEqual(
            cfg.to_dict(with_all=True),
            {
                "includes": [],
                "test": {
                    "some_map": {"field1": 1, "field2": 2},
                    "complex_sub": {"includes": [], "sub": {"subscalar": -1}},
                    "enum_value": "ZERO",
                    "enum_value_no_default": None,
                    "scalar": -1.0,
                    "scalar_no_default": None,
                    "sub": {"subscalar": -1},
                    "sub2": {"subscalar": -1},
                    "bool_field_no_default": False,
                    "bool_field_with_default_false": False,
                    "bool_field_with_default_true": True,
                    "launcher": {},
                },
            },
        )

    def testToStrWithDefault(self):
        root_cfg = self.data_dir / "root.prototxt"
        cfg = heyhi.conf.load_proto_message(root_cfg)
        EXPECTED_MSG = """
test {
  scalar: -1.0
  sub {
    subscalar: -1
  }
  sub2 {
    subscalar: -1
  }
  enum_value: ZERO
  complex_sub {
    sub {
      subscalar: -1
    }
  }
  bool_field_with_default_false: false
  bool_field_with_default_true: true
}
        """.strip()
        self.assertEqual(cfg.to_str_with_defaults().strip(), EXPECTED_MSG)

    def testGetters(self):
        root_cfg = self.data_dir / "root.prototxt"
        cfg = heyhi.conf.load_proto_message(root_cfg)
        # No magic fields before to_frozen.
        with self.assertRaises(AttributeError):
            cfg.test.oneof_field
        cfg = cfg.to_frozen()
        # Oneof is not set.
        self.assertEqual(cfg.test.oneof_field, None)
        self.assertEqual(cfg.test.which_oneof_field, None)
        self.assertEqual(cfg.test.WhichOneof("oneof_field"), None)
        # Oneof values set to Nones of not set
        self.assertEqual(cfg.test.oneof_value1, None)
        # Read doesn't change selection.
        self.assertEqual(cfg.test.oneof_field, None)
        self.assertEqual(cfg.test.which_oneof_field, None)
        self.assertEqual(cfg.test.WhichOneof("oneof_field"), None)

        # Cannot change.
        with self.assertRaises(RuntimeError):
            cfg.test.oneof_value1 = 11
        cfg = cfg.to_editable()
        cfg.test.oneof_value1 = 11
        cfg = cfg.to_frozen()
        self.assertEqual(cfg.test.oneof_field, 11)
        self.assertEqual(cfg.test.which_oneof_field, "oneof_value1")
        self.assertEqual(cfg.test.WhichOneof("oneof_field"), "oneof_value1")

    def test_wrap_cfg(self):
        cfg = conf.conf_pb2.TestTask(
            scalar_no_default=42,
            oneof_value1=11,
            sub=conf.conf_pb2.TestTask.SubMessage(subscalar=9),
            complex_sub=dict(
                includes=[{"path": "p1", "mount": "m1"}, {"path": "p2", "mount": "m2"}]
            ),
        )

        cfg = cfg.to_frozen()

        self.assertEqual(cfg.scalar_no_default, 42)
        print(type(cfg))
        self.assertIsNone(cfg.enum_value_no_default)

        # cfg.to_dict() returns explicitly set keys only
        self.assertEqual(
            set(cfg.to_dict().keys()), {"complex_sub", "scalar_no_default", "oneof_value1", "sub"}
        )

        self.assertEqual(cfg.WhichOneof("oneof_field"), "oneof_value1")
        self.assertEqual(cfg.which_oneof_field, "oneof_value1")
        self.assertEqual(cfg.oneof_field, cfg.oneof_value1)

        # uninitialized submessage within oneof is None
        self.assertIsNone(cfg.oneof_value2)
        self.assertIsNone(cfg.oneof_value3)
        # uninitialized submessage outside oneof is not None
        self.assertIsNotNone(cfg.sub2)
        self.assertEqual(cfg.sub2.subscalar, -1)

        # cfg is read-only
        with self.assertRaises(RuntimeError):
            cfg.scalar = 5

        # lists and dicts
        self.assertTrue(isinstance(cfg.complex_sub.includes, tuple))
        self.assertEqual(len(cfg.complex_sub.includes), 2)
        self.assertEqual(cfg.complex_sub.includes[0].path, "p1")
        self.assertEqual(cfg.complex_sub.includes[1].path, "p2")

        # enums
        self.assertEqual(cfg.enum_value, "ZERO")
        self.assertEqual(cfg.enum_value_no_default, None)

    def testFrozenConstructor(self):
        # Creating a message using frozen class constructor
        msg = conf.conf_cfgs.TestTask(
            scalar=10, sub=conf.conf_pb2.TestTask.SubMessage(subscalar=11)
        )
        self.assertTrue(msg.is_frozen())
        self.assertEqual(msg.scalar, 10)
        self.assertEqual(msg.to_dict(), {"scalar": 10, "sub": {"subscalar": 11}})

    def testFrozenPickle(self):
        root_cfg = self.data_dir / "root.prototxt"
        cfg = heyhi.conf.load_proto_message(root_cfg)
        unpacked = pickle.loads(pickle.dumps(cfg.to_frozen()))
        self.assertEqual(str(cfg), str(unpacked))

    def testFrozenRepr(self):
        root_cfg = self.data_dir / "root.prototxt"
        cfg = heyhi.conf.load_proto_message(root_cfg)
        self.assertEqual(repr(cfg), repr(cfg.to_frozen()))


class TestConvWithOverrides(unittest.TestCase):
    def setUp(self):
        self.data_dir = pathlib.Path(__file__).parent / "data"

    def testToStrWithDefault(self):
        root_cfg = self.data_dir / "root.prototxt"
        cfg = heyhi.conf.load_proto_message(root_cfg).to_frozen()
        cfg = heyhi.conf.conf_with_overrides(
            cfg, overrides=["test.scalar=3.0", "test.sub2.subscalar=5"]
        )
        EXPECTED_MSG = """
test {
  scalar: 3.0
  sub {
    subscalar: -1
  }
  sub2 {
    subscalar: 5
  }
  enum_value: ZERO
  complex_sub {
    sub {
      subscalar: -1
    }
  }
  bool_field_with_default_false: false
  bool_field_with_default_true: true
}
        """.strip()
        self.assertEqual(cfg.to_str_with_defaults().strip(), EXPECTED_MSG)
