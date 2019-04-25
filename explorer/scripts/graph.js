/* global vis, document, window, WebSocket, moment, $ */

'use strict';

const globalContext = {
    _privates: {},
    _now: moment(),
};

const nodes = new vis.DataSet([]);
const edges = new vis.DataSet([]);
const times = new vis.DataSet([]);

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
        }, 250);
    }


    /*
    document.querySelectorAll('p').forEach((p) => {
        p.style.color = event.target.value;
    });
    */
};

// eslint-disable-next-line no-unused-vars
const toggleExpand = function () {
    const dropdown = $('#nodeWeight');
    const related = $('body').find('[aria-labelledby="nodeWeight"]');
    if (dropdown.hasClass('show')) {
        dropdown.removeClass('show');
        related.removeClass('show');
    } else {
        dropdown.addClass('show');
        related.addClass('show');
    }
};

// eslint-disable-next-line no-unused-vars
const getDropdownContent = function (event) {
    const selected = $(event.target).html();
    const selectedVal = $(event.target).attr('name');
    $('#nodeWeight').html(selected);
    toggleExpand();

    sendMessageJson({
        cmd: 'change_node_weight',
        value: selectedVal,
    });
};

const onOpen = function (content) {
    console.log('connected');
    globalContext.websocket = {
        content,
        connection: true,
    };

    $('#serverConnection').css('color', '#00f100');
    $('#serverConnection').attr('title', 'server connected');
};

const onClose = function () {
    console.log('disconnected');
    delete globalContext.websocket;

    // eslint-disable-next-line no-use-before-define
    setTimeout(doConnect, 1000);
    $('#serverConnection').css('color', 'red');
    $('#serverConnection').attr('title', 'server disconnected');
};

const onError = function (evt) {
    console.log(`error: ${evt.data}`);

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

const onMessage = function (evt) {
    console.log('get message');
    const msg = JSON.parse(evt.data);

    const options = {
        editable: {
            add: false,
            updateTime: true,
        },
        onMove: (item, callback) => {
            globalContext._privates.daterange = {
                start: item.start,
                end: item.end,
            };
            sendMessageJson({
                cmd: 'timeline_seek',
                time: item.start.getTime(),
            });
            timeline.focus(1);
            item.start = moment(item.start);
            item.end = moment(item.end);

            callback(item);
        },
        onMoving: (item, callback) => {
            resizeDateRangePicker({
                start: moment(item.start.getTime()),
                end: moment(item.end.getTime()),
            });

            callback(item);
        },
        // min: moment(),
        // max: moment(),
        zoomMin: 600000,
        zoomMax: 314496000000,
    };
    let current;
    let start;
    let end;

    const containerTimeline = document.getElementById('mytimeline');
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

            while (containerTimeline.firstChild) {
                containerTimeline.removeChild(containerTimeline.firstChild);
            }

            times.clear();
            times.add([
                {
                    id: 1,
                    start,
                    end, // end is optional
                    content: 'selected period',
                    style: 'color: white; background-color: #17a2b8; border-color: #0c7e90; text-align: center;',
                    type: 'range',
                    editable: {
                        add: true,
                        updateTime: true,
                        updateGroup: false,
                        remove: false,
                        overrideItems: true,
                    },
                },
                {
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
                },
            ]);

            timeline = new vis.Timeline(containerTimeline, times, options);
            // timeline.setItems(times);
            timeline.setSelection(1);
            timeline.focus(1);

            timeline.on('doubleClick', (properties) => {
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
                    start,
                    end,
                });

                timeline.focus(1);

                sendMessageJson({
                    cmd: 'timeline_seek',
                    time: startRescal.unix() * 1000,
                });
            });

            timeline.on('click', () => {
                timeline.setSelection(1);
            });

            sendMessageJson({
                cmd: 'timeline_seek',
                time: start.unix() * 1000,
            });

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

            // FIXME: Stabilize?
            break;

        case 'network_update':
            console.log('network_update:');
            console.log(msg.deleted_nodes.length, 'deleted nodes');
            console.log(msg.deleted_edges.length, 'deleted edges');
            console.log(msg.nodes.length, 'nodes');
            console.log(msg.edges.length, 'edges');
            console.log('mul =', msg.mul);

            edges.remove(msg.deleted_edges);
            nodes.remove(msg.deleted_nodes);

            edges.forEach((edge) => {
                edge.weight *= msg.mul;
            });

            nodes.update(msg.nodes);
            edges.update(msg.edges);

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
        globalContext._privates.daterange = {
            start,
            end,
        };
        sendMessageJson({
            cmd: 'timeline_seek',
            time: start / 1,
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
            },
        },
    };

    const container = document.getElementById('mynetwork');
    network = new vis.Network(container, data, options);
}

const init = function () {
    initDateRangePicker();
    initColorPicker();
    initNetwork();

    setTimeout(doConnect, 1000);

    console.log('finished graph.init()');
};

window.addEventListener('load', init, false);
