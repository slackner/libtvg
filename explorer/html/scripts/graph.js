/* global vis, document, window, WebSocket, moment, $ */

'use strict';

const globalContext = {
    _privates: {},
    _now: moment(),
    disableElements: [],
};

const nodes = new vis.DataSet([]);
const edges = new vis.DataSet([]);
const times = new vis.DataSet([
    {
        id: 1,
        start: globalContext._now.startOf('hour'),
        end: globalContext._now.startOf('hour'), // end is optional
        style: 'color: white; background-color: #7ca7af; border-color: #33636b; text-align: center;',
        content: 'space of time',
        type: 'range',
        editable: {
            updateTime: true,
            updateGroup: false,
            remove: false,
        },
    },
]);

// const WebSocket = require('ws');


let network; // eslint-disable-line no-unused-vars
let timeline;
let websocket;

const sendMessageJson = function (message) {
    if (!websocket) {
        console.log('missing connection to server');
    } else {
        websocket.send(JSON.stringify(message));
    }
};

const watchColorPicker = function (event) {
    const flag = event.target.id.split('-')[1];
    const color = event.target.value;

    sendMessageJson({
        cmd: 'save_custom_color',
        color,
        flag,
    });

    const recolorSet = [];
    nodes.forEach((item) => {
        if (flag === item.nodeType) {
            recolorSet.push({
                id: item.id,
                color,
            });
        }
    });
    nodes.update(recolorSet);
};

// eslint-disable-next-line no-unused-vars
const changeEdgeWeight = function (event) {
    const selected = $(event.target).html();
    const selectedVal = $(event.target).attr('name');
    $('#edgeWeight').html(selected);

    sendMessageJson({
        cmd: 'change_edge_weight',
        value: selectedVal,
    });

    $('#loading').show();
};

// eslint-disable-next-line no-unused-vars
const changeNodeSize = function (event) {
    const selected = $(event.target).html();
    const selectedVal = $(event.target).attr('name');
    $('#nodeSize').html(selected);

    sendMessageJson({
        cmd: 'change_node_size',
        value: selectedVal,
    });

    $('#loading').show();
};

const updateTimeline = function (end) {
    if (document.getElementById('liveMonitoring').checked) {
        // Update grey background and selected area
        const dateRangePicker = times.get(1);
        const dateRangePickerWidth = dateRangePicker.end - dateRangePicker.start;
        const start = (end - dateRangePickerWidth);

        times.update([
            {
                id: 1,
                start,
                end,
            },
            {
                id: 2,
                end,
            },
        ]);

        // eslint-disable-next-line no-use-before-define
        resizeDateRangePicker({
            start: moment(start),
            end: moment(end),
        });

        timeline.focus(1);

        sendMessageJson({
            cmd: 'timeline_seek',
            start,
            end,
        });
        $('#loading').show();
    } else {
        // Update only grey backround
        times.update([
            {
                id: 2,
                end,
            },
        ]);
    }
};

const onOpen = function (/* event */) {
    console.log('connected');
    globalContext.connected = true;

    $('#serverConnection').css('color', '#00f100');
    $('#serverConnection').attr('title', 'server connected');

    globalContext.disableElements.forEach((id) => {
        $(`#${id}`).attr('disabled', false);
    });
};

const onClose = function (/* event */) {
    console.log('disconnected');
    globalContext.connected = false;
    websocket = undefined;

    $('#serverConnection').css('color', 'red');
    $('#serverConnection').attr('title', 'server disconnected');

    globalContext.disableElements.forEach((id) => {
        $(`#${id}`).attr('disabled', true);
    });

    times.update({
        id: 1,
        style: 'color: white; background-color: #7ca7af; border-color: #33636b; text-align: center;',
    });

    // eslint-disable-next-line no-use-before-define
    setTimeout(doConnect, 10000);
};

const onError = function (error) {
    console.log(`error: ${JSON.stringify(error)}`);

    websocket.close();
};

const resizeDateRangePicker = function (item) {
    const drp = $('input[name="daterange"]').data('daterangepicker');
    if (item.start) {
        drp.setStartDate(item.start);
    }
    if (item.end) {
        drp.setEndDate(item.end);
    }
};

const initColorPicker = function (context) {
    const listElement = document.getElementById('colorizeList');

    while (listElement.firstChild) {
        const removed = listElement.removeChild(listElement.firstChild);

        globalContext.disableElements = globalContext.disableElements.filter((key) => {
            if (key === removed.querySelector('input').id) {
                return false;
            }

            return true;
        });
    }

    $.each(context.nodeTypes, (type, element) => {
        const divRow = document.createElement('div');
        divRow.classList.add('row');

        const divCol8 = document.createElement('div');
        divCol8.classList.add('col-md-8');

        const divCol4 = document.createElement('div');
        divCol4.classList.add('col-md-4');
        divCol4.setAttribute('style', 'text-align: right;');

        const i = document.createElement('i');
        i.className = element.class;
        i.setAttribute('data-placement', 'left');

        const title = document.createElement('span');
        title.setAttribute('style', 'font-weight: normal; padding-left: 10px;');
        title.innerHTML = element.title;

        const input = document.createElement('input');
        input.setAttribute('id', `nodeColor-${type}`);
        input.setAttribute('type', 'color');
        input.setAttribute('value', element.color);
        input.className = 'nodeColor';
        globalContext.disableElements.push(`nodeColor-${type}`);

        const text = document.createElement('b');
        text.innerHTML = type;

        listElement.appendChild(divRow);
        divRow.appendChild(divCol8);
        divCol8.appendChild(i);
        divCol8.appendChild(input);
        divCol8.appendChild(title);
        divRow.appendChild(divCol4);
        divCol4.appendChild(text);

        const node = document.querySelector(`#nodeColor-${type}`);
        node.addEventListener('change', watchColorPicker, false);
    });
};

const onMessage = function (event) {
    console.log('get message');
    const msg = JSON.parse(event.data);

    let current;
    let start;
    let end;

    switch (msg.cmd) {
        case 'timeline_set_options':
            console.log('timeline_set_options');

            // Seek to the center of the given interval.
            current = new Date((msg.min + msg.max) / 2);
            start = moment(current).subtract(1, 'hours').startOf('hours');
            end = moment(current).startOf('hours');

            resizeDateRangePicker({
                start,
                end,
            });

            times.update({
                id: 1,
                start,
                end,
                style: 'color: white; background-color: #17a2b8; border-color: #0c7e90; text-align: center;',
            });

            times.update({
                id: 2,
                start: moment(msg.min),
                end: moment(msg.max),
                type: 'background',
                style: 'background-color: #eee;',
                editable: {
                    updateTime: false,
                    updateGroup: false,
                    remove: false,
                },
            });

            timeline.redraw();
            timeline.setSelection(1);
            timeline.focus(1);

            sendMessageJson({
                cmd: 'timeline_seek',
                start: start.unix() * 1000,
                end: end.unix() * 1000,
            });
            $('#loading').show();
            break;

        case 'set_context':
            console.log('set_context:');
            console.log(msg.context);

            initColorPicker(msg.context);
            break;

        case 'network_set':
            console.log('network_set:');
            console.log(msg.nodes.length, 'nodes');
            console.log(msg.edges.length, 'edges');

            nodes.clear();
            edges.clear();

            nodes.add(msg.nodes);
            edges.add(msg.edges);

            $('#initNodes').html(msg.nodes.length);
            $('#initEdges').html(msg.edges.length);
            $('#loading').hide();

            // FIXME: Stabilize?
            break;

        case 'focus_timeline':
            console.log('focus_timeline');

            timeline.focus(1);
            break;

        case 'update_timeline':
            console.log('update_timeline');
            updateTimeline(msg.max);
            break;

        default:
            console.log(msg.cmd);
            console.log('response:', msg.data);
            break;
    }
};

// eslint-disable-next-line no-unused-vars
const doConnect = function () {
    websocket = new WebSocket('ws://localhost:8000/');
    websocket.onopen = onOpen;
    websocket.onclose = onClose;
    websocket.onmessage = onMessage;
    websocket.onerror = onError;
};

const initDateRangePicker = function () {
    const nowHour = globalContext._now.startOf('hour');
    $('input[name="daterange"]').daterangepicker({
        timePicker: true,
        timePicker24Hour: true,
        startDate: nowHour,
        endDate: nowHour,
        opens: 'left',
        locale: {
            format: 'DD.MM.YYYY HH:mm',
        },
    }, (start, end) => {
        sendMessageJson({
            cmd: 'timeline_seek',
            start: start / 1,
            end: end / 1,
        });
        $('#loading').show();

        times.update({
            id: 1,
            start: moment(start).startOf('seconds'),
            end: moment(end).startOf('seconds'), // end is optional
        });
        timeline.focus(1);
        /*
        itemData.start = moment(start);
        itemData.end = moment(end);
        timeline.itemSet.items[0].setData(itemData);
        timeline.itemSet.items[0].repositionX();
        timeline.itemsData.update(itemData);
        */

        // console.log(`A new date selection was made: ${start.format('YYYY-MM-DD')} to ${end.format('YYYY-MM-DD')}`);
    });
};

const initNetwork = function () {
    const data = {
        nodes,
        edges,
    };

    const options = {
        nodes: {
            shape: 'dot',
            scaling: {
                min: 2,
                max: 30,
                label: {
                    enabled: true,
                },
            },
            font: {
                size: 16,
                face: 'Tahoma',
            },
        },
        edges: {
            color: {
                inherit: 'both',
            },
            // width: 0.15,
            smooth: {
                type: 'continuous',
            },
            scaling: {
                min: 2,
                max: 30,
            },
        },
        interaction: {
            hideEdgesOnDrag: true,
            tooltipDelay: 200,
        },
        physics: {
            stabilization: false,
            barnesHut: {
                gravitationalConstant: -10000,
                springConstant: 0.002,
                springLength: 150,
                avoidOverlap: 1,
            },
        },
    };

    const container = document.getElementById('mynetwork');
    network = new vis.Network(container, data, options);
};

const initTimeline = function () {
    const options = {
        start: new Date('2018'),
        end: new Date('2020'),
        editable: {
            add: false,
            updateTime: true,
        },
        zoomMin: 600000,
        zoomMax: 314496000000,
        onMove: (item, callback) => {
            if (globalContext.connected) {
                sendMessageJson({
                    cmd: 'timeline_seek',
                    start: item.start.getTime(),
                    end: item.end.getTime(),
                });
                $('#loading').show();

                timeline.focus(1);
                item.start = moment(item.start);
                item.end = moment(item.end);

                callback(item); // send back adjusted item
            } else {
                callback(null); // cancel updating the item
            }
        },
        onMoving: (item, callback) => {
            if (globalContext.connected) {
                resizeDateRangePicker({
                    start: moment(item.start.getTime()),
                    end: moment(item.end.getTime()),
                });

                callback(item); // send back adjusted item
            } else {
                callback(null); // cancel updating the item
            }
        },
    };

    const container = document.getElementById('mytimeline');
    timeline = new vis.Timeline(container, times, options);
    // timeline.setItems(times);

    timeline.on('doubleClick', (properties) => {
        if (globalContext.connected) {
            const item = times.get(1);
            const timeOffset = item.end.diff(item.start) / 2;
            const startRescal = moment(properties.time).subtract({ ms: timeOffset });
            const endRescal = moment(properties.time).add({ ms: timeOffset });

            times.update({
                id: 1,
                start: startRescal.startOf('seconds'),
                end: endRescal.startOf('seconds'), // end is optional
            });

            resizeDateRangePicker({
                start: startRescal,
                end: endRescal,
            });

            timeline.focus(1);

            sendMessageJson({
                cmd: 'timeline_seek',
                start: startRescal.unix() * 1000,
                end: endRescal.unix() * 1000,
            });
            $('#loading').show();
        }
    });

    timeline.on('click', () => {
        timeline.setSelection(1);
    });
};

const floatString = function (num) {
    if (Number.isInteger(num)) {
        return `${num}.0`;
    }

    return num.toString();
};

// eslint-disable-next-line no-unused-vars
const downloadSnapshot = function () {
    const lines = [];
    const pos = network.getPositions();

    lines.push('graph');
    lines.push('[');
    lines.push('  directed 0');

    nodes.forEach((node) => {
        lines.push('  node');
        lines.push('  [');
        lines.push(`    id ${node.id}`);
        lines.push(`    label "${node.label}"`);
        lines.push(`    weight ${floatString(node.value)}`);
        lines.push('    graphics');
        lines.push('    [');
        lines.push(`      x ${floatString(pos[node.id].x)}`);
        lines.push(`      y ${floatString(-pos[node.id].y)}`);
        lines.push('    ]');
        lines.push('  ]');
    });

    edges.forEach((edge) => {
        lines.push('  edge');
        lines.push('  [');
        lines.push(`    source ${edge.from}`);
        lines.push(`    target ${edge.to}`);
        lines.push(`    weight ${floatString(edge.value)}`);
        lines.push('  ]');
    });

    lines.push(']');

    const base64 = window.btoa(lines.join('\n'));

    const link = document.createElement('a');
    link.href = `data:application/octet-stream;base64,${encodeURI(base64)}`;
    link.download = `snapshot_${moment().format('YYYY-MM-DD_HH-mm')}.gml`;
    link.click();
};

const init = function () {
    initDateRangePicker();
    initNetwork();
    initTimeline();

    globalContext.disableElements.push('liveMonitoring');
    globalContext.disableElements.push('dropdownEdgeWeight');
    globalContext.disableElements.push('dropdownNodeSize');
    globalContext.disableElements.push('downloadGML');
    globalContext.disableElements.push('daterangepicker');

    setTimeout(doConnect, 1000);

    console.log('initialized');
};

window.addEventListener('load', init, false);
