#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the APGLv3 license found in the
# LICENSE file in the fairdiplomacy_external directory of this source tree.
#
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import collections
import functools
import html
import uuid

import jinja2

import conf.misc_cfgs
from fairdiplomacy import pydipcc
from fairdiplomacy.models.consts import POWERS, LOCS
from fairdiplomacy_external.ui import map_renderer


def get_header_stuff() -> str:
    """Returns HTML to be embedded in the head of an HTML document (scripts, inline css, etc)
    that enables the rest of the functions in this render module to function properly
    """
    return r"""
<script
  src="https://code.jquery.com/jquery-3.5.1.min.js"
  integrity="sha256-9/aliU8dGd2tb6OSsuzixeV4y/faTqgFtohetphbbj0="
  crossorigin="anonymous">
</script>
<script
  src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"
  integrity="sha256-T0Vest3yCU7pafRw9r+settMBX6JkKN06dqBnpQ8d30="
  crossorigin="anonymous">
</script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.4.1/semantic.min.css" integrity="sha512-8bHTC73gkZ7rZ7vpqUQThUDhqcNFyYi2xgDgPDHc+GXVGHXq+xPjynxIopALmOPqzo9JZj0k6OqqewdGO3EsrQ==" crossorigin="anonymous" referrerpolicy="no-referrer" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.4.1/semantic.min.js" integrity="sha512-dqw6X88iGgZlTsONxZK9ePmJEFrmHwpuMrsUChjAw1mRUhUITE5QU9pkcSox+ynfLhL15Sv2al5A0LVyDCmtUw==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>

<!-- TAGS START -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/tagify/4.1.2/jQuery.tagify.min.js" integrity="sha512-7FC+s51N+76MQGvN5fMvu2zpVb1iO06wJGohUVD0n/Z58xLwXS2ssTbW7ZzT/NFjaNOW7QyKH2GxkCYOxL3doA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tagify/4.1.2/tagify.min.css" integrity="sha512-PmDlF8+NwANjpP2Kc1EqXOUCtV0bSz1mr5HC9zii3CxtHPDKGHsioVf+JFBqLItNJol0S9A+TXIjnG626ZxUHQ==" crossorigin="anonymous" referrerpolicy="no-referrer" />
<!-- TAGS END -->
<style>
  .dip_message, .dip_order, .gameNav, #messagesFilter, .dip_game, .section {
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji";
  }
  .vizTitle {
    font-size: 2.5rem;
    font-weight:500;
  }
  h3 {
    font-size: 1.5rem;
    font-weight:500;
    margin-bottom: 5px;
  }
  hr {
    height:1px;
  }
  .gameNav a {
    color: #007bff;
    text-decoration: none;
    background-color: transparent;
  }
  .gameNav a:hover {
    text-decoration: underline;
  }
  .gameNav {
    position: relative;
    flex-wrap: wrap;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.0rem 1rem;
    width: 100%;
    max-width: 1000px;
    background-color: #e3f2fd;
  }
  .section {
    display:flex;
    flex-flow: column;
  }

  .messagesTitle {
    margin-top: 10px;
  }
  .messageLabel {
    font-weight:bold;
  }
  .dip_message {
    margin-top:0;
    margin-bottom: 0.5rem;
  }
  .dip_order {
    margin-top:0;
    margin-bottom: 0.5rem;
  }
  p.dip_message {
    margin-top:0;
    margin-bottom: 0.5rem;
  }
  p.dip_order {
    margin-top:0;
    margin-bottom: 0.5rem;
  }
  #messagesFilter {
    max-width:22vh;
    background-color: #fde3f2;
    padding: 5px;
    width: 100%;
    overflow: hidden;
    display:flex;
    flex-flow:column;
    border:outset;
    border-color:pink;
  }
  .messagesFilterFixed {
    position: fixed;
    top: 100px;
    right: 10px;
    z-index:100;
  }
  .formOption {
    white-space:nowrap;
  }
  .scrollBox {
    overflow:auto;
    border:ridge;
    padding:3px;
    max-height: 650px;
  }
  .hidden {
    display: none !important;
  }
  .RUSSIA {
    color: #555555 !important;
  }
  .ENGLAND {
    color: #9900BB !important;
  }
  .FRANCE {
    color: #0033DD !important;
  }
  .GERMANY {
    color: #997766 !important;
  }
  .TURKEY {
    color: #999922 !important;
  }
  .AUSTRIA {
    color: #BB0000 !important;
  }
  .ITALY {
    color: #009900 !important;
  }
  .highlightedLoc {
    font-weight:bold;
  }
  .highlightedLoc.mapLoc {
    font-size:2.4em !important;
    stroke-width:2.4 !important;
    stroke:#224455 !important;
    fill:#66FFFF !important;
  }
  .addTest {
      display: none;
  }
  BODY.loading .addTest {
      /* hide buttons until JS kicks in */
      display: none !important;
  }
  .dip_message:hover .addTest {
      display: inline;
  }
  .dip_order:hover .addTest {
      display: inline;
  }
  .field.inline .checkbox {
    padding-left: 10px;
  }
  .phase-dropdown {
    display: inline-block;
    position: relative;
  }
  .phase-dropdown > span {
    border-bottom: 1px dashed blue;
  }
  .phase-dropdown-content {
    display: none;
    position: absolute;
    padding: 10px;
    left: 0px;
    top: 20px;
    width: 300px;
    background: white;
    border: 1px solid black;

  }
  .phase-dropdown:hover .phase-dropdown-content  {
    display: block;
  }
  .messages_and_map {
    display: flex;
    flex-flow: column
  }
  @media only screen and (min-width: 1800px) {
    .messages_and_map {
      flex-flow: row;
    }
  }


</style>
<script>
  function getRadioValue(form_id, def) {
    form = document.getElementById(form_id)
    ret = def;
    if(form) {
      for (i = 0; i < form.length; i++) {
        if (form[i].checked) {
          ret = form[i].value;
        }
      }
    }
    return ret;
  }
  function updateMessages() {
    var power1 = getRadioValue("power1_form", "ALL");
    var power2 = getRadioValue("power2_form", "ALL");
    var num_powers = (power1 != "ALL") + (power2 != "ALL");
    var power = (power1 == "ALL") ? power2 : power1;

    var msgs = document.getElementsByClassName("dip_message");
    if(!msgs)
      return;
    for (i = 0; i < msgs.length; i++) {
      var msg = msgs[i];
      var hide = false;
      var sender = msg.getAttribute("sender");
      var recipient = msg.getAttribute("recipient");
      if (sender != power1 && recipient != power1 && power1 != "ALL" && recipient != "ALL") {
        hide = true;
      }
      if (sender != power2 && recipient != power2 && power2 != "ALL" && recipient != "ALL") {
        hide = true;
      }

      if (hide) {
        msg.classList.add("hidden");
      } else {
        msg.classList.remove("hidden");
      }
    }
  }
  function goToPhase(game_dom_id, phase_id, phase_name) {
    var gameElt = document.getElementById(game_dom_id);
    var phaseElts = gameElt.getElementsByClassName("dip_phase");
    var found = false;
    for (var i = 0; i < phaseElts.length; i++) {
      var phaseElt = phaseElts[i];
      var hide = phaseElt.getAttribute("phase") != phase_id || phase_id == "all";
      //Make sure we aren't hiding everything!
      if(!hide) {
        found = true;
        break;
      }
    }
    if(found) {
      for (var i = 0; i < phaseElts.length; i++) {
        var phaseElt = phaseElts[i];
        var hide = phaseElt.getAttribute("phase") != phase_id || phase_id == "all";
        if (hide) {
          phaseElt.classList.add("hidden");
        } else {
          phaseElt.classList.remove("hidden");
        }
      }
      const urlParams = new URLSearchParams(window.location.search);
      urlParams.set('phase', phase_name);
      window.history.replaceState(null,"","/?"+urlParams);
    }
  }

  function highlightWord(id, word) {
    var elt = document.getElementById(id);
    var wordElts = document.getElementsByClassName("highlightedLoc");
    while(wordElts.length) {
      wordElts[0].classList.remove("highlightedLoc");
    }
    var wordElts = document.getElementsByClassName("loc_"+word);
    for (var i = 0; i < wordElts.length; i++) {
      wordElts[i].classList.add("highlightedLoc");
    }
  }
  var _saveTestSituation = null;
  // Specify a function f to save test situations, instead of simply displaying them in the html for user to
  // copy somewhere. f should accept a string name, a json object that is the test situation json, and callback g,
  // and it should call g with a string message to display upon error or success of saving.
  function setSaveTestSituation(f) {
    _saveTestSituation = f;
  }

  function objectifyForm(formArray) {
    //serialize data function
    var returnArray = {};
    for (var i = 0; i < formArray.length; i++){
        returnArray[formArray[i]['name']] = formArray[i]['value'];
        if (formArray[i]['name'] == "tags") {
          var value_dict = eval(formArray[i]['value']);
          var value_list = [];
          for (var j in value_dict) value_list.push(value_dict[j].value);
          returnArray[formArray[i]['name']] = value_list;
        }
    }
    return returnArray;
  }

  function makeTestSituation(event) {
    event.preventDefault();

    const testSituationJson = objectifyForm($("#testSituationSaverDiv form").serializeArray());
    delete testSituationJson["language"];
    console.log(testSituationJson);

    if(_saveTestSituation) {
      var show_message = function(text, classname) {
        if (text.responseText) text = text.responseText;
        $("#testSituationSaverDiv .output").html("<div />");
        $("#testSituationSaverDiv .output div").addClass(classname).html(text);
      }
      _saveTestSituation(
        name,
        testSituationJson,
        function(response) {show_message(response, "ui positive message");},
        function(response) {show_message(response, "ui negative message");}
      );
      updateCommonTags();
    }
    else {
      warning = "// Copy this into test_situations.json or another file of test situations.\\n// Please make sure the game_path is correct and refers to a permanent location everyone can see in /checkpoint";
      result = {};
      result[name] = testSituationJson;
      json = JSON.stringify(result, null, 2);
      document.getElementById(testSituationId+"_output").textContent = warning + "\\n" + json;
    }
    return false;
  }

  function findRemoteParentWithClass(el, parrent_class) {
    while (el && el[0].tagName != "BODY") {
      el = el.parent();
      if (el.hasClass(parrent_class)) return el;
    }
  }
  function updateCommonTags() {
    $.get("/get_tags/", function(result){
      $('#testSituationSaverDiv #commonTags').html("");
      for (var i in result.result) {
        var tagName = result.result[i];
        $('#testSituationSaverDiv #commonTags').append(
          $("<a href='#' class='ui label'>" + tagName + "</a>").click(function(event){
            event.preventDefault();
            var tagName = $(this).text();
            $('#testSituationSaverDiv input[name=tags]').data('tagify').addTags([tagName]);
          })
        );
      }
    });
  }

  $(function() {
    $("body").removeClass("loading");
    $("#messagesFilter").draggable();
    updateMessages();
    $(".addTest").click(function (event) {
        event.preventDefault();
        var wrapper = $(this).parent();
        $("#testSituationSaverDiv form")[0].reset();
        $('#testSituationSaverDiv input[name=tags]').data('tagify').removeAllTags();
        $("#testSituationSaverDiv .output").text("");
        $("#testSituationSaverDiv").appendTo(wrapper).show();
        $("#testSituationSaverDiv input[name=phase]").val(
          findRemoteParentWithClass(wrapper, "dip_phases").attr("phase_name")
        );
        var pov_power, time_sent;
        if (wrapper.hasClass("dip_message")) {
          time_sent = wrapper.attr("time_sent");
          pov_power = wrapper.attr("sender");
        } else {
          time_sent = -1;
          pov_power = wrapper.attr("power");
        }
        $("#testSituationSaverDiv input[name=time_sent]").val(time_sent);
        $("#testSituationSaverDiv input[name=pov_power]").each(function(){
          this.checked = this.value == pov_power;
        })
        var $game_path = $("#testSituationSaverDiv input[name=game_path]");
        if (!$game_path.val()) {
          //Heuristic guess for direct html files - take the browser's view of the file path
          //and replace html -> json
          $game_path.val(location.pathname.replace(/\.html*/, ".json"))
        }
    });
    $('#testSituationSaverDiv>form').submit(makeTestSituation);
    $("#testSituationSaverDiv button[name=close]").click(function(){
        event.preventDefault();
        $("#testSituationSaverDiv").hide();
        $('#testSituationSaverDiv').form("reset");
    });
    var $tagInput = $('#testSituationSaverDiv input[name=tags]').tagify({
      duplicates: false
    });
    updateCommonTags();
  });
</script>
    """


@functools.lru_cache(None)
def _compile(tpl: str) -> jinja2.Template:
    return jinja2.Template(tpl)


def _annotation_content_to_html(
    content: conf.misc_cfgs.AnnotatedGame.Annotation.AnnotationContent,
):
    if content.plain_text is not None:
        return content.plain_text.replace("\n", "<br/>")
    elif content.message_list is not None:
        message_list_html = "<br/>".join(
            f"""
            <i>
            <b>(CF)</b>
            <span class="messageLabel">
                <span class="{m.sender}">{m.sender}</span>
                ->
                <span class="{m.recipient}">{m.recipient}</span>
            </span> ({m.timestamp}):
            {m.content}
            </i>
            """
            for m in content.message_list.messages
        )
        return message_list_html
    else:
        return "???"


def _canonize_message(
    msg: Dict[str, Any],
    message_annotations: Dict[
        int, List[conf.misc_cfgs.AnnotatedGame.Annotation.AnnotationContent]
    ],
) -> Dict[str, Any]:
    new_msg = {}

    def try_copy_key(key, as_key):
        if as_key not in new_msg:
            if isinstance(msg, dict) and key in msg:
                new_msg[as_key] = msg[key]
            if hasattr(msg, key):
                new_msg[as_key] = getattr(msg, key)

    # Handle parlai-style messages as well as old dip game style messages
    try_copy_key("message", as_key="message")
    try_copy_key("sender", as_key="sender")
    try_copy_key("recipient", as_key="recipient")
    try_copy_key("speaker", as_key="sender")
    try_copy_key("time_sent", as_key="time_sent")
    try_copy_key("listener", as_key="recipient")
    try_copy_key("time_sent", as_key="time_sent")
    if "sender" in new_msg:
        new_msg["sender"] = new_msg["sender"].upper()
    if "recipient" in new_msg:
        new_msg["recipient"] = new_msg["recipient"].upper()

    new_msg["annotation_list"] = message_annotations.get(msg["time_sent"], [])
    # annotation is a simplified rendering of annotations. Mostly for game2html.
    # React will render annotations in the frontend
    if new_msg["annotation_list"]:
        new_msg["annotation"] = _annotation_content_to_html(new_msg["annotation_list"][0])
        new_msg["annotation_list"] = [x.to_dict() for x in new_msg["annotation_list"]]
    else:
        new_msg["annotation"] = None
    return new_msg


def canonize_messages(
    messages: Sequence[Dict[str, Any]], annotations: Optional[conf.misc_cfgs.AnnotatedGame],
) -> List[Dict[str, Any]]:
    message_annotations = collections.defaultdict(list)
    if annotations is not None:
        for annotation in annotations.annotations:
            if annotation.message_at and annotation.message_at > 0:
                message_annotations[annotation.message_at].append(annotation.content)

    return [_canonize_message(x, message_annotations) for x in messages]


def render_message_list(
    title: str,
    messages: Sequence[Dict[str, Any]],
    annotations: Optional[conf.misc_cfgs.AnnotatedGame],
    show_add_test: bool = False,
    show_timestamps: bool = False,
) -> str:
    """Render into html a list of message from a diplomacy game.

    Parameters:
    title (str): Title for the list of messages
    messages: e.g. game.get_all_phases()[phase_id].messages.values()

    Returns: str.
    """
    template = """
    <div class="section">
      {% if title %}
        <h3 class="messagesTitle">{{ title }}:</h3>
      {% endif %}

      <div class="scrollBox">
        {% for msg in messages %}
          <p class="dip_message" time_sent="{{msg.time_sent}}" sender="{{msg.sender}}" recipient="{{msg.recipient}}" title="{{msg.time_sent}}">
            <span class="messageLabel">
              <span class="{{msg.sender}}">{{msg.sender}}</span>
                ->
              <span class="{{msg.recipient}}">{{msg.recipient}}</span>
            </span>
            {% if show_time_sent %}({{msg.time_sent}}){% endif %}: {{msg.message}}
            {% if show_add_test %} <a href="#" class="addTest">+test</a> {% endif %}
            {% if msg.annotation %}
              <br/><span class="msg_annotation">{{ msg.annotation|safe }}</span><br/>
            {% endif %}
          </p>

        {% else %}
          (No messages)
        {% endfor %}
      </div>
    </div>
    """
    messages = canonize_messages(messages, annotations)
    template = _compile(template)
    return template.render(title=title, messages=messages)


def render_message_filterer(fixed=False, filter1=False) -> str:
    """Render into html a panel that can be used to filter all messages by power.
    Returns: str
    """
    fixed_class = "messagesFilterFixed" if fixed else ""
    template = """
<div class="menu {{fixed_class}}" id="messagesFilter">
  <form id="power1_form">
  <b>Filter to sender or receiver{{'' if filter1 else ' 1'}}:</b>
  <div>
  <span class="formOption">
  <input type="radio" name="power1" id="filter1ALL" value="ALL" onclick="updateMessages()" checked/>
    <label for="filter1ALL">ALL</label>
  </span>
  {% for power in POWERS %}
  <span class="formOption">
  <input type="radio" name="power1" id="filter1{{power}}" value="{{power}}" onclick="updateMessages()"/>
    <label class="{{power}}" for="filter1{{power}}">{{power}}</label>
  </span>
  {% endfor %}
  </div>
  </form>
  {% if not filter1 %}
  <form id="power2_form">
  <b>Filter to sender or receiver 2:</b>
  <div>
  <span class="formOption">
  <input type="radio" name="power2" id="filter2ALL" value="ALL" onclick="updateMessages()" checked/>
    <label for="filter2ALL">ALL</label>
  </span>
  {% for power in POWERS %}
  <span class="formOption">
  <input type="radio" name="power2" id="filter2{{power}}" value="{{power}}" onclick="updateMessages()"/>
    <label class="{{power}}" for="filter2{{power}}">{{power}}</label>
  </span>
  {% endfor %}
  </div>
  </form>
  {% endif %}

</div>
    """
    template = _compile(template)
    return template.render(**locals(), POWERS=POWERS)


def render_orders(
    title: str, orders: Dict[str, List[str]], section_id: Optional[str] = None
) -> str:
    """Render into html a dictionary of orders by power from a diplomacy game.

    Parameters:
    title (str): Title for the list of orders
    orders: A dictionary mapping power -> list of orders for that power
    section_id (optional str): If provided, when a location is hovered, will highlight all occurrences of location in the
        DOM elt with this id.

    Returns: str"""
    template = """
{% if title %}
<h3 class="messagesTitle">{{title}}:</h3>
{% endif %}
<div class="dip_order_block">
    joint_action = {
{% for pwr, order_html in order_htmls.items() %}
  <p class="dip_order" power="{{pwr}}"><span class="messageLabel {{pwr}}">'{{pwr}}'</span> : {{order_html|safe}}, <a href="#" class="addTest">+test</a></p>
{% endfor %}
    }
</div>
    """
    template = _compile(template)
    all_pieces = {
        power: [order.split(" ") for order in power_orders]
        for (power, power_orders) in orders.items()
    }

    def strip_coast(location):
        slash_idx = location.find("/")
        if slash_idx != -1:
            location = location[:slash_idx]
        return location

    def find_power_for_loc(location):
        location = strip_coast(location)
        for power in all_pieces:
            power_pieces = all_pieces[power]
            for pieces in power_pieces:
                if len(pieces) >= 2 and location == pieces[1]:
                    return power
        return None

    order_htmls = {}
    for power in all_pieces:
        power_pieces = all_pieces[power]
        power_order_htmls = []
        for pieces in power_pieces:
            order_html_pieces = []
            if len(pieces) < 3:
                order_html_pieces = [html.escape(piece) for piece in pieces]
            else:

                def wrap_order_piece(piece):
                    escaped = html.escape(piece)
                    coastless = strip_coast(escaped)
                    # For board locations, wrap in a fancy span that highlights on mouseover.
                    if section_id is not None and coastless in LOCS:
                        coastless = strip_coast(escaped)
                        return f"""<span class="loc_{coastless}" onmouseover="highlightWord('{section_id}','{coastless}')" onmouseout="highlightWord('{section_id}','')">{escaped}</span>"""
                    else:
                        return escaped

                order_html_pieces.append(wrap_order_piece(pieces[0]))
                order_html_pieces.append(
                    f"""<span class="{power}">{wrap_order_piece(pieces[1])}</span>"""
                )
                order_html_pieces.append(wrap_order_piece(pieces[2]))
                for piece in pieces[3:]:
                    other_power = find_power_for_loc(piece)
                    if other_power is not None:
                        order_html_pieces.append(
                            f"""<span class="{other_power}">{wrap_order_piece(piece)}</span>"""
                        )
                    else:
                        order_html_pieces.append(wrap_order_piece(piece))
            order_html = " ".join(order_html_pieces)
            power_order_htmls.append("'" + order_html + "'")
        power_order_htmls = "[" + ",".join(power_order_htmls) + "]"
        order_htmls[power] = power_order_htmls

    return template.render(title=title, order_htmls=order_htmls)


def render_test_situation_saver(game_json_path: Optional[str] = None) -> str:
    test_situation_id = uuid.uuid4()
    game_json_path = "" if game_json_path is None else game_json_path
    template = """
<div id="testSituationSaverDiv" class="ui message" style="display: none">
  <form class="ui mini form">

  <div class="inline field">
  <label for="{{test_situation_id}}_name">Name:</label>
  <input id="{{test_situation_id}}_name" type="text" size="20" name="name" placeholder="my_new_test_situation"/>
  <small class="helper">Unique name. Will be autogenerated if empty.</small>
  </div>

  <input id="{{test_situation_id}}_phase" type="hidden" name="phase" value="????"/>
  <input id="{{test_situation_id}}_time_sent" type="hidden" name="time_sent" value="????"/>
  <input id="{{test_situation_id}}_game_json_path" type="hidden" name="game_path" value="{{game_json_path}}"/>

  <div class="inline field">
  <label for="{{test_situation_id}}_comment">Comment:</label>
  <input id="{{test_situation_id}}_comment" type="text" size="120" name="comment" placeholder="A sentence about why this test situation is interesting"/>
  </div>

  <div class="inline field">
  <label for="{{test_situation_id}}_tags">Tags:</label>
  <input id="{{test_situation_id}}_tags" data-role="tagsinput" type="text" size="50" name="tags" />
  <div id="commonTags"></div>
  </div>

  <div class="field inline">
  <label>PoV:</label>
  {% for power in POWERS %}
  <span class="field inline">
  <div class="ui radio checkbox">
  <input type="radio" name="pov_power" id="{{test_situation_id}}_{{power}}" class="{{test_situation_id}}_power" value="{{power}}"/>
  <label class="{{power}}" for="{{test_situation_id}}_{{power}}">{{power}}</label>
  </div>
  </span>
  {% endfor %}
  </div>

  <div class="field">
  <input class="ui primary button submit" type="submit" value="Make" />
  <button class="ui button" name="close">Close</button>
  </div>
  </form>
  <pre class="output"></pre>
</div>
    """
    template = jinja2.Template(template)
    return template.render(
        game_json_path=game_json_path, test_situation_id=test_situation_id, POWERS=POWERS,
    )


def _render_phase_contents(
    phase_name,
    image,
    messages,
    orders,
    logs: Optional[Sequence[str]],
    annotations: Optional[conf.misc_cfgs.AnnotatedGame],
    game_json_path: Optional[str],
) -> str:
    section_id = str(uuid.uuid4())
    logs_id = str(uuid.uuid4())
    template = """
<div class="section" id="{{section_id}}">
<div class="messages_and_map">
{{ rendered_message_list|safe }}
{{ image|safe }}
</div>
{{ order_html|safe }}
{% if logs %}
<b>Logs <a href="#" onClick="$('#{{logs_id}}').toggle(); return false;">Show/Hide</a></b>
<pre id="{{logs_id}}" style="display: none">{{ logs|e }}</pre>
{% endif %}

{{ test_situation_saver|safe }}
</div>
    """

    template = _compile(template)
    if messages:
        rendered_message_list = render_message_list("Messages", messages, annotations)
    else:
        rendered_message_list = ""

    if logs is not None:
        logs = "\n".join(logs)

    order_html = render_orders("Orders", orders, section_id=section_id)
    situation_check_saver = render_test_situation_saver(game_json_path)
    return template.render(
        section_id=section_id,
        image=image,
        rendered_message_list=rendered_message_list,
        order_html=order_html,
        logs=logs,
        logs_id=logs_id,
        phase_name=phase_name,
    )


def _render_nav(game_dom_id, phase_id, phase, phase_list, all_url):
    template = """
  <nav class="navbar navbar-light gameNav">
    <a href="?" class="vizTitle">Diplomacy viz</a>
    {% if num_phases %}
    Phase: {{ phase_id + 1 }} / {{ num_phases }}
      <div class="phase-dropdown">
        <span>({{phase}})</span>
        <div class="phase-dropdown-content">
        <div>
        {%for i in range(num_phases)%}
        {% if i > 0 and i < num_phases - 1 and phase_names[i - 1][1:5] != phase_names[i][1:5] %}</div><div>{% endif %}
        {% if i == phase_id %}
            {{phase_names[i]}}
        {% else %}
        <a href="javascript:void(0)"
            onclick='goToPhase({{ game_dom_id|tojson}}, {{ i|tojson }}, {{ phase_names[i]|tojson }}); return false;'>
            {{phase_names[i]}}</a>
        {% endif %}
        {% endfor %}
        </div>
        </div>
      </div>
      {% if all_url %}
          <a href="{{all_url|safe}}">
            all
          </a>
      {% endif %}
      {% for phase_link in phase_links %}
        {% if phase_link.active %}
          <a href="javascript:void(0)"
             onclick='goToPhase({{ game_dom_id|tojson}}, {{ phase_link.phase_id|tojson }}, {{ phase_link.phase_name|tojson }}); return false;'>
            {{ phase_link.txt }}
          </a>
        {% else %}
          <span>{{ phase_link.txt }}</span>
        {% endif %}
      {% endfor %}
    {% endif %}
  </nav>
    """

    num_phases = len(phase_list)
    phase_links = []

    def make_phase_link(offset, txt):
        new_id = min(max(phase_id + offset, 0), num_phases - 1)
        active = new_id != phase_id
        return {
            "active": active,
            "txt": txt,
            "phase_id": new_id,
            "phase_name": str(phase_list[new_id]),
        }

    phase_links.append(make_phase_link(-100000000, "first"))
    phase_links.append(make_phase_link(-10, "prev x 10"))
    phase_links.append(make_phase_link(-1, "prev"))
    phase_links.append(make_phase_link(1, "next"))
    phase_links.append(make_phase_link(10, "next x 10"))
    phase_links.append(make_phase_link(100000000, "last"))

    template = _compile(template)
    return template.render(
        phase_id=phase_id,
        phase=phase,
        num_phases=num_phases,
        phase_names=phase_list,
        phase_links=phase_links,
        game_dom_id=game_dom_id,
        all_url=all_url,
    )


def render_game(
    game_name: str,
    game: pydipcc.Game,
    initial_phase_id: int = 0,
    all_url: Optional[str] = None,
    image_generator: Optional[Callable] = None,
    annotations: Optional[conf.misc_cfgs.AnnotatedGame] = None,
    game_json_path: Optional[str] = None,
) -> str:
    """Render a game into html, with navigable sections for every phase.

    Parameters:
      game_name (str): Displayed as a title
      game pydipcc.Game: The game to display.
      initial_phase_id (int): The initial phase to be displayed.
      all_url (optional string): If provided, a url to provide as an "all" link in the nav.
      image_generator: callable phase->html; will be used to produce game images.
      annotations: additional annotations for the game.
      game_json_path (str, optional): If provided, will be used to fill in situation check field.

    Returns: str
    """
    rendered_phases = []
    phase_names = game.get_all_phase_names()
    game_dom_id = "GAME_" + str(game_name)
    for phase in phase_names:
        phase_id = phase_names.index(phase)
        if image_generator is None:
            image = map_renderer.render(game, phase)
        else:
            image = image_generator(phase)
        messages = list(game.get_all_phases()[phase_id].messages.values())
        orders = game.get_all_phases()[phase_id].orders
        rendered_nav = _render_nav(game_dom_id, phase_id, phase, phase_names, all_url)

        # If the game passed in was a pydipcc game with logs, also render that
        logs = None
        if hasattr(game, "get_logs"):
            game_logs = game.get_logs()
            if phase in game_logs:
                logs = game_logs[phase]

        rendered_phase_contents = _render_phase_contents(
            phase, image, messages, orders, logs, annotations, game_json_path,
        )

        if phase_id == initial_phase_id:
            maybe_hidden = ""
        else:
            maybe_hidden = " hidden"

        template = """
          <div class="section dip_phases dip_phase{{ maybe_hidden }}" phase="{{phase_id}}" phase_name="{{ phase }}">
            {{ rendered_nav|safe }}
            {{ rendered_phase_contents|safe }}
          </div>
        """
        template = _compile(template)
        rendered_phase = template.render(
            maybe_hidden=maybe_hidden,
            phase_id=phase_id,
            phase=phase,
            rendered_nav=rendered_nav,
            rendered_phase_contents=rendered_phase_contents,
        )
        rendered_phases.append(rendered_phase)

    rendered_game_contents = " ".join(rendered_phases)
    test_situation_saver = render_test_situation_saver(game_json_path)

    template = """
       {{ test_situation_saver|safe }}
       <div class="dip_game" id="{{game_dom_id}}">
         <!-- Hack to get url(#id) to work when there are multiple svgs and some are hidden -->
         <svg height="0">
         <marker id="arrow" markerHeight="4" markerUnits="strokeWidth" markerWidth="4" orient="auto" refX="5" refY="5" viewBox="0 0 10 10">
           <path d="M 0 0 L 10 5 L 0 10 z"/>
         </marker>
         </svg>
         {{ rendered_game_contents|safe }}
       </div>
    """
    template = _compile(template)
    return template.render(
        game_dom_id=game_dom_id,
        rendered_game_contents=rendered_game_contents,
        test_situation_saver=test_situation_saver,
    )


def render_phase(
    game: pydipcc.Game,
    phase: Optional[Union[str, int]] = None,
    annotations: Optional[conf.misc_cfgs.AnnotatedGame] = None,
    game_json_path: Optional[str] = None,
) -> str:
    """Render a single phase into html.

    Note: supports both old python diplomacy Game and pydipcc Game, but newer features
    like (logs) are only be supported in pydipcc.

    Parameters:
    game (fairdiplomacy.game.Game or pydipcc.Game): The game to display
    phase (str or int, optional): Name or index of phase in game, like "W1903A"
        or 3 or -1. If not specified, uses the current phase.
    game_json_path (str, optional): If provided, will be used to fill in game path for generating
        test situations. Defaults to a guess based on URL.

    Returns: str
    """
    # Convert to old python diplomacy Game since that's necessary for rendering the image
    phase_names = game.get_all_phase_names()
    if type(phase) == int:
        try:
            phase = str(phase_names[phase])  # type: ignore
        except IndexError:
            template = """<div class="section dip_phase"> Invalid phase: {{phase}} </div>"""
            template = _compile(template)
            return template.render(phase=phase)

    if phase is None:
        phase = phase_names[-1]
    assert isinstance(phase, str)

    phase_id = phase_names.index(phase)
    messages = list(game.get_all_phases()[phase_id].messages.values())
    orders = game.get_all_phases()[phase_id].orders

    image = (
        f'<svg style="min-height: 700px; min-width:930px">{map_renderer.render(game, phase)}</svg>'
    )
    # If the game passed in was a pydipcc game with logs, also render that
    logs = None
    if hasattr(game, "get_logs"):
        game_logs = game.get_logs()
        if phase in game_logs:
            logs = game_logs[phase]

    rendered_phase_contents = _render_phase_contents(
        phase, image, messages, orders, logs, annotations, game_json_path
    )

    template = """
      <div class="section dip_phase" phase="{{phase_id}}">
        {{ rendered_phase_contents|safe }}
      </div>
    """
    template = _compile(template)
    return template.render(phase_id=phase_id, rendered_phase_contents=rendered_phase_contents)
