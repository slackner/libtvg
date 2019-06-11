/* global vis, document, window, WebSocket, moment, $ */

'use strict';

const globalContext = {
    _privates: {},
    _now: moment(),
};

const nodes = new vis.DataSet([]);
const edges = new vis.DataSet([]);
const times = new vis.DataSet([
    {
        id: 1,
        start: globalContext._now.startOf('hour'),
        end: globalContext._now.startOf('hour'), // end is optional
        style: 'color: white; background-color: #17a2b8; border-color: #0c7e90; text-align: center;',
        content: 'selected period',
        type: 'range',
        editable: {
            add: true,
            updateTime: true,
            updateGroup: false,
            remove: false,
            overrideItems: true,
        },
    },
]);

const settings = {
    color: [
        {
            title: 'location',
            flag: 'LOC',
            class: 'mr-3 far fa-compass',
            color: '#00ffff',
        },
        {
            title: 'organisation',
            flag: 'ORG',
            class: 'mr-3 fas fa-globe',
            color: '#0040ff',
        },
        {
            title: 'actor',
            flag: 'ACT',
            class: 'mr-3 far fa-user-circle',
            color: '#8000ff',
        },
        {
            title: 'date',
            flag: 'DAT',
            class: 'mr-3 far fa-clock',
            color: '#ff0080',
        },
        {
            title: 'term',
            flag: 'TER',
            class: 'mr-3 fas fa-exclamation-circle',
            color: '#80ff00',
        },
    ],
};

// const WebSocket = require('ws');


let network; // eslint-disable-line no-unused-vars
let timeline;
let websocket;

const sendMessageJson = function (message) {
    if (!websocket) {
        console.log('no serverconnection established still');
    } else {
        websocket.send(JSON.stringify(message));
    }
};

const watchColorPicker = function (event) {
    globalContext._privates.nodeColor = event.target.value;

    if (!globalContext.watchDog) {
        globalContext.watchDog = true;

        setTimeout(() => {
            delete globalContext.watchDog;
            const flag = event.target.id.split('-')[1];

            sendMessageJson({
                cmd: 'recolor_graph_nodes',
                color: globalContext._privates.nodeColor,
                flag,
            });

            $('#loading').show();
        }, 250);
    }


    /*
    document.querySelectorAll('p').forEach((p) => {
        p.style.color = event.target.value;
    });
    */
};

// eslint-disable-next-line no-unused-vars
const getDropdownContent = function (event) {
    const selected = $(event.target).html();
    const selectedVal = $(event.target).attr('name');
    $('#nodeWeight').attr('value', selected);

    sendMessageJson({
        cmd: 'change_node_weight',
        value: selectedVal,
    });

    $('#loading').show();
};

const onOpen = function (/* event */) {
    console.log('connected');
    globalContext.connected = true;

    $('#serverConnection').css('color', '#00f100');
    $('#serverConnection').attr('title', 'server connected');
    $('#daterangepicker').attr('disabled', false);
};

const onClose = function (/* event */) {
    console.log('disconnected');
    globalContext.connected = false;
    websocket = undefined;

    $('#serverConnection').css('color', 'red');
    $('#serverConnection').attr('title', 'server disconnected');
    $('#daterangepicker').attr('disabled', true);

    // eslint-disable-next-line no-use-before-define
    setTimeout(doConnect, 1000);
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
            });

            times.update({
                id: 2,
                start: moment(msg.min),
                end: moment(msg.max),
                type: 'background',
                style: 'background-color: #eee;',
                editable: {
                    add: false,
                    updateTime: true,
                    updateGroup: false,
                    remove: false,
                    overrideItems: true,
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

            // FIXME: Apply context.
            break;

        case 'network_set':
            console.log('network_set:');
            console.log(msg.nodes.length, 'nodes');
            console.log(msg.edges.length, 'edges');

            nodes.clear();
            edges.clear();

            nodes.add(msg.nodes);
            edges.add(msg.edges);

            $('#loading').hide();

            // FIXME: Stabilize?
            break;

        case 'focus_timeline':
            console.log('focus_timeline');

            timeline.focus(1);
            break;

        default:
            console.log(msg.cmd);
            console.log('response:', evt.data);
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

        $('#loading').show();

        // console.log(`A new date selection was made: ${start.format('YYYY-MM-DD')} to ${end.format('YYYY-MM-DD')}`);
    });
};

const initColorPicker = function () {
    const listElement = document.getElementById('colorizeList');
    settings.color.forEach((element) => {
        const divRow = document.createElement('div');
        divRow.classList.add('row');

        const divCol = document.createElement('div');
        divCol.classList.add('col-md-12');

        const i = document.createElement('i');
        i.className = element.class;
        i.setAttribute('data-toggle', 'tooltip');
        i.setAttribute('data-placement', 'top');
        i.setAttribute('title', element.title);

        const input = document.createElement('input');
        input.setAttribute('id', `nodeColor-${element.flag}`);
        input.setAttribute('type', 'color');
        input.setAttribute('value', element.color);
        input.className = 'nodeColor';


        listElement.appendChild(divRow);
        divRow.appendChild(divCol);
        divCol.appendChild(i);
        divCol.appendChild(input);

        const node = document.querySelector(`#nodeColor-${element.flag}`);
        node.addEventListener('change', watchColorPicker, false);
    });
}

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
}

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
                timeline.focus(1);
                item.start = moment(item.start);
                item.end = moment(item.end);

                $('#loading').show();

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
            const startRescal = moment(properties.snappedTime).subtract({ ms: timeOffset });
            const endRescal = moment(properties.snappedTime).add({ ms: timeOffset });

            times.update({
                id: 1,
                start: startRescal.startOf('seconds'),
                end: endRescal.startOf('seconds'), // end is optional
            });

            resizeDateRangePicker({
                startRescal,
                endRescal,
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

const init = function () {
    initDateRangePicker();
    initColorPicker();
    initNetwork();
    initTimeline();

    setTimeout(doConnect, 1000);

    console.log('finished graph.init()');
};

window.addEventListener('load', init, false);
