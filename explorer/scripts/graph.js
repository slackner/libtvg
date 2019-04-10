/* global vis, document, window, WebSocket, $ */

'use strict';

const globalContext = {};

const nodes = new vis.DataSet([]);
const edges = new vis.DataSet([]);
const times = new vis.DataSet([]);

const settings = {
    color: [
        {
            title: 'location',
            flag: 'L',
            class: 'mr-3 far fa-compass',
            color: '#00ffff',
        },
        {
            title: 'organisation',
            flag: 'O',
            class: 'mr-3 fas fa-globe',
            color: '#0040ff',
        },
        {
            title: 'actor',
            flag: 'A',
            class: 'mr-3 far fa-user-circle',
            color: '#8000ff',
        },
        {
            title: 'date',
            flag: 'D',
            class: 'mr-3 far fa-clock',
            color: '#ff0080',
        },
        {
            title: 'term',
            flag: 'T',
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
    globalContext._privates = {
        nodeColor: event.target.value,
    };

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

const onMessage = function (evt) {
    console.log('get message');
    const msg = JSON.parse(evt.data);
    const options = {};
    let current;

    switch (msg.cmd) {
        case 'timeline_set_options':
            console.log('timeline_set_options:');

            options.min = new Date(msg.min);
            options.max = new Date(msg.max);
            options.clickToUse = true;

            // Seek to the center of the given interval.
            current = new Date((msg.min + msg.max) / 2);
            timeline.setCustomTime(current, 't1');
            timeline.setCustomTimeTitle('preset', 't1');
            timeline.setOptions(options);

            sendMessageJson({
                cmd: 'timeline_seek',
                time: current.getTime(),
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

        default:
            console.log(msg.cmd);
            console.log('response:', evt.data);
            break;
    }
};

// eslint-disable-next-line no-unused-vars
const doConnect = function () {
    console.log('websocket established');
    websocket = new WebSocket('ws://localhost:8000/');
    websocket.onopen = onOpen;
    websocket.onclose = onClose;
    websocket.onmessage = onMessage;
    websocket.onerror = onError;
};

const init = function () {
    // Initialize network

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

    const data = {
        nodes,
        edges,
    };

    const optionsNetwork = {
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

    const containerNetwork = document.getElementById('mynetwork');
    network = new vis.Network(containerNetwork, data, optionsNetwork);

    // Initialize timeline
    const optionsTimeline = {
        // showCurrentTime: true,
        start: new Date(Date.now() - (1000 * 60 * 60 * 24)),
        end: new Date(Date.now() + (1000 * 60 * 60 * 24 * 6)),
        editable: false,
    };

    const containerTimeline = document.getElementById('mytimeline');
    timeline = new vis.Timeline(containerTimeline, times, optionsTimeline);
    timeline.addCustomTime(new Date(), 't1');
    timeline.on('timechanged', (properties) => {
        if (properties.id === 't1') {
            sendMessageJson({
                cmd: 'timeline_seek',
                time: properties.time.getTime(),
            });
        }
    });

    setTimeout(doConnect, 1000);

    console.log('finished graph.init()');
};

window.addEventListener('load', init, false);
