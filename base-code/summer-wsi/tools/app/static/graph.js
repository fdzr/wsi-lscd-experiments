const colors = ['orange', 'blue', 'red', 'black', 'purple', 'green']
let summaryTableContainer = document.getElementById('substs-panel');

const drawSingleEdgeSelectionSummaryTable = (edge_id) => {
    let distance = dataset.distances[edge_id]
    let distanceDiv = document.createElement('div')
    distanceDiv.innerHTML = '<u>Distance</u>: ' + distance.toExponential(2);
    let nodeIds = edge_id.split('_').slice(1)

    let substsSection = createSection('Combined substitutes with frequencies')
    let substs = getMostFrequentSubsts(nodeIds)

    substs.forEach((substFreqPair) => {
        let subst = substFreqPair[1]
        let freq = substFreqPair[0]
        let b = document.createElement('b')
        let span = document.createElement('span')
        b.innerHTML = subst
        substsSection.append(b)
        substsSection.append(': ')
        span.innerHTML = freq
        substsSection.append(span)
        substsSection.append(document.createElement('br'))
    })

    let div = document.createElement('div')
    div.style.position = 'relative';
    div.style.border = 'solid 1px #cacaca';
    div.style.background = '#fff';
    div.style.padding = '5px';
    div.style.width = 'calc(100% - 10px)';
    div.style.height = '350px';
    div.style.borderRadius = '3px';
    div.style.overflow = 'scroll';
    div.style.zIndex = '1000';
    div.style.marginBottom = '10px';

    div.append(distanceDiv)
    div.append(substsSection)

    return div
}

const drawMultiEdgeSelectionSummaryTable = (edge_ids) => {

}

const getGraphData = () => {
    return fetch(
        '/experiment/' + experiment_id + '/graph/' + word + '/data',
        {
            method: 'GET'
        }
    )
        .then((response) => response.json())
        .then((result) => {
            dataset = result;
            return result
        });
}

const drawGraph = () => {
    if (Object.keys(dataset).length === 0) {
        redraw()
    } else {
        getEmbeddings().then(
            (embeddings) => {
                animate(embeddings, embeddings.length - 1)
            })
    }
}

const getEmbeddings = () => {
    toggleProcessingOverlay()
    return fetch(
        '/experiment/' + experiment_id + '/graph/' + word + '/embeddings'
        + '?' + new URLSearchParams({
            method: document.getElementById('method').value,
            perplexity: document.getElementById('perplexity').value,
            n_iter: document.getElementById('n_iter').value,
            exaggeration: document.getElementById('exaggeration').value,
            early_exaggeration: document.getElementById('early_exaggeration').value,
            early_exaggeration_n_iter: document.getElementById('early_exaggeration_n_iter').value,
            metric: document.getElementById('metric').value
        }),
        {
            method: 'GET',
        }
    ).then((response) => response.json()).then((result) => {
        return result
    });
}

const updateStyle = () => {

    window.cy.style().selector('node').style(
        'background-color', function (node) {
            return colors[node.data(document.getElementById('color_field').value)]
        }
    ).selector('node:selected').style(
        'background-color', '#55f'
    ).update()
}

const sliderChange = (target) => {
    let text = document.getElementById(target + '-label').innerHTML
    let value = document.getElementById(target).value

    document.getElementById(target + '-label').innerHTML = text.split(' ').slice(0, -1).join(' ') + ' ' + value
}

const animate = (embeddings, j) => {
    if (j >= embeddings.length) return;

    window.cy.nodes().layout({
        name: 'preset',
        animate: true,
        animationDuration: 1000,
        fit: {
            eles: window.cy.nodes(),
            padding: 10
        },
        stop: () => {
            if (embeddings.length - 1 > 0) {
                animate(embeddings, j + embeddings.length - 1)
            }
            toggleProcessingOverlay()
        },
        transform: (node) => {
            let position = {};
            position.x = embeddings[j][parseInt(node.data('id'))][0];
            position.y = embeddings[j][parseInt(node.data('id'))][1];
            return position;
        }
    }).run()
}

const onMethodChange = (target) => {
    if (target.value === 'tsne') {
        document.getElementById('tsne-params').style.display = 'block'
    } else {
        document.getElementById('tsne-params').style.display = 'none'
    }
}

const selectOneNode = (nodeId) => {
    let neighbours = getNearestNeighbours(nodeId, 10)

    neighbours.forEach((node) => {
        let edge_id = getEdgeId(node[1], nodeId)
        let existingEdge = window.cy.elements('edge[id="' + edge_id + '"]')
        if (existingEdge.length === 0) {
            window.cy.add({
                style: {
                    label: node[0].toExponential(2)
                },
                data: {
                    id: edge_id,
                    source: nodeId,
                    target: node[1]
                }
            });
        }
    });
}

const selectMultipleNodes = (nodeIds) => {
    window.cy.elements('edges').remove();
    if (nodeIds.length > 10) return;
    nodeIds.forEach((node1_id) => {
        nodeIds.forEach((node2_id) => {
            if (node1_id !== node2_id) {
                let edge_id = getEdgeId(node1_id, node2_id)
                let edge = window.cy.elements('edge[id="' + edge_id + '"]');
                let distance = dataset.distances[edge_id].toExponential(2)
                if (edge.length > 0) {
                    edge.css({
                        content: distance + ""
                    })
                } else {
                    window.cy.add({
                        style: {
                            label: distance
                        },
                        data: {
                            id: edge_id,
                            source: node1_id,
                            target: node2_id
                        }
                    });
                }
            }
        })
    });
}

const selectNode = () => {
    const selectedNodes = window.cy.elements('node:selected');
    if (selectedNodes.length === 1) {
        let nodeId = selectedNodes[0].data('id');
        selectOneNode(nodeId);
        summaryTableContainer.replaceChildren(drawSingleNodeSelectionSummaryTable(nodeId));
    } else {
        let nodeIds = selectedNodes.map((node) => {
            return node.data('id')
        })
        selectMultipleNodes(nodeIds);
        summaryTableContainer.replaceChildren(drawMultiNodeSelectionSummaryTable(nodeIds));
    }
}

const selectOneEdge = (edge_id) => {

}

const selectMultipleEdges = (edge_ids) => {

}

const selectEdge = () => {
    const selectedEdges = window.cy.elements('edge:selected');

    if (selectedEdges.length === 1) {
        let edge_id = selectedEdges[0].data('id');
        selectOneEdge(edge_id);
        summaryTableContainer.replaceChildren(drawSingleEdgeSelectionSummaryTable(edge_id));
    } else {
        let edge_ids = selectedEdges.map((edge) => {
            return edge.data('id')
        })
        selectMultipleEdges(edge_ids);
        summaryTableContainer.replaceChildren(drawMultiEdgeSelectionSummaryTable(edge_ids));
    }

}

const initCy = () => {
    window.cy = cytoscape({
        container: document.getElementById('cy'),
        draggable: true,
        boxSelectionEnabled: true,

        elements: {
            nodes: dataset.elements.nodes.map((node) => {
                node.position = {x: 0, y: 0}
                return node
            }), edges: []
        },

        style: [ // the stylesheet for the graph
            {
                selector: 'node',
                style: {
                    'background-color': function (node) {
                        return colors[node.data(document.getElementById('color_field').value)]
                    },
                    'shape': 'ellipse',
                    'width': '10',
                    'height': '10'
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'background-color': '#55f',
                }
            },
            {
                selector: 'edge',
                style: {
                    'line-color': '#ccc',
                    'curve-style': 'bezier',
                    'width': 4,
                    'font-size': 5,
                }
            }, {
                selector: 'edge:selected',
                style: {
                    'line-color': '#55f',
                    'width': 6,
                    'font-size': 10,
                }
            }
        ]
    });

    window.cy.on('select', 'edge', (evt) => {
        selectEdge(evt.target.id)
    });

    window.cy.on('unselect', 'edge', (evt) => {
        evt.target.remove()
        summaryTableContainer.replaceChildren('');
    });

    window.cy.on('unselect', 'node', (evt) => {
        const selectedNodes = window.cy.$('node:selected');
        summaryTableContainer.replaceChildren(drawMultiNodeSelectionSummaryTable(selectedNodes.map((node) => {
            return node.data('id')
        })));

        setTimeout(() => {
            evt.target.connectedEdges().forEach((edge) => {
                if (!edge.selected()) {
                    edge.remove()
                }
            })
        }, 100)

    });

    window.cy.on('mouseover', 'node', (evt) => {
        let container = document.getElementById('cy');
        let tooltip = createToolTip(evt.target.data('id'), evt.renderedPosition.x, evt.renderedPosition.y)
        container.append(tooltip);
    });

    window.cy.on('mouseout', 'node', (evt) => {
        document.getElementById('tooltip').remove();
    });

    window.cy.on('select', 'node', function (evt) {
        selectNode(evt.target.id)
    });
}

const redraw = () => {
    getGraphData().then(() => {
        toggleProcessingOverlay()

        setTimeout(() => {
            document.getElementById('cy').remove();
            let container = document.getElementById('container');
            let cyDiv = document.createElement('div')
            cyDiv.id = 'cy';
            container.append(cyDiv);

            initCy()

            toggleProcessingOverlay()
        }, 100);
    }).then(() => {
        getEmbeddings().then((embeddings) => {
            animate(embeddings, embeddings.length - 1)
        })
    });
}

document.addEventListener('DOMContentLoaded', function () {
    drawGraph()
});

