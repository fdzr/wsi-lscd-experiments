const colors = ['orange', 'blue', 'red', 'black', 'purple', 'green']
let summaryTableContainer = document.getElementById('substs-panel');
const cont = d3.select('#svgContainer');
let nodes = null;
let links = null;
let texts = null;
let selectedNodeId = null;
let dragged = false;
let clusterCircleSize = 80;
let nodeCircleSize = 30;
let defaultFontSize = '40px'
let selectedFontSize = '60px'
let n_clusters = parseInt(document.getElementById('n_clusters').value)

document.addEventListener('DOMContentLoaded', () => {
    toggleProcessingOverlay()
    getDendrogramData().then(() => {
        drawDendrogram()
        nClustersSliderChange("n_clusters", updateCircleSizes)
        toggleProcessingOverlay()
    });
});

const drawDendrogram = () => {

    d3.select("#svgImage").remove()

    let svg = cont.append('svg').attr("id", "svgImage").attr("height", Object.keys(dataset.data).length * 150).attr("width", Object.keys(dataset.data).length * 100);
    let width = +svg.attr("width"),
        height = +svg.attr("height"),
        g = svg.append("g").attr("transform", "translate(0,0)");

    svg.on('click', function () {
        if (!dragged)
            selectNode(null);
    })

    let cluster = d3.cluster()
        .size([height, width - 160]);

    let root = d3.hierarchy(dataset.tree);

    cluster(root);

    links = g.selectAll(".link")
        .data(root.descendants().slice(1))
        .enter().append("path")
        .attr("class", "link")
        .style("stroke-width", '5px')
        .attr("d", diagonal);

    nodes = g.selectAll(".node")
        .data(root.descendants())
        .enter().append("g")
        .style('cursor', 'pointer')
        .attr("class", function (d) {
            return "node" + (isFinalCluster(n_clusters, d.data.id) ? ' cluster-node ' : '') + (d.children ? " node--internal" : " node--leaf");
        })
        .attr("transform", function (d) {
            return "translate(" + d.y + "," + d.x + ")";
        });

    nodes.append("circle")
        .attr("r", function (d) {
            return isFinalCluster(n_clusters, d.data.id) ? clusterCircleSize : nodeCircleSize
        }).on("click", function (d) {
        setTimeout(() => {
            selectNode(d.data.id);
        }, 100)
        return false;
    })
        .style("fill", function (d) {
            return typeof d.data.cluster === 'undefined' ? 'black' : colors[d.data.cluster];
        });

    texts = nodes.append("text")
        .attr("dy", 3)
        .attr("id", (d) => d.data.id)
        .attr("x", function (d) {
            return d.children ? -4 : 4;
        }).on('click', (d) => {
            setTimeout(() => {
                selectNode(d.data.id);
            }, 100);
        })
        .style("text-anchor", function (d) {
            return d.children ? "end" : "start";
        })
        .style('cursor', 'pointer')
        .style('font-size', defaultFontSize)
        .style("fill", function (d) {
            return typeof d.data.cluster === 'undefined' ? 'black' : colors[d.data.cluster];
        });

    texts.append('tspan').text(function (d) {
        return d.data.name === null ? "" : d.data.name.split(d.data.word_in_context)[0];
    });

    texts.append('tspan').style('font-weight', 'bold').text(function (d) {
        return d.data.name === null ? "" : d.data.word_in_context;
    });

    texts.append('tspan').text(function (d) {
        return d.data.name === null ? "" : d.data.name.split(d.data.word_in_context).slice(1).join(d.data.word_in_context);
    });

    texts.insert("title").text(function (d) {
        return d.data.context;
    });

    nodes.append('text')
        .attr('class', 'dist-text')
        .attr("id", (d) => d.data.id + '-dist')
        .style("text-anchor", 'start')
        .attr('dy', '11px')
        .attr('dx', '120px')
        .on('click', (d) => {
            summaryTableContainer.replaceChildren(drawClusterDistancesTable(selectedNodeId, d.data.id))
            summaryTableContainer.append(drawDiscriminativeSubsts(selectedNodeId, d.data.id))
            d3.selectAll('text.dist-text').style('font-weight', 'normal')
            d3.select('text[id="' + d.data.id + '-dist"]').style('font-weight', 'bold')
            d3.event.stopPropagation()
        })

    function diagonal(d) {
        return "M" + d.y + "," + d.x
            + "C" + (d.parent.y + 100) + "," + d.x
            + " " + (d.parent.y + 100) + "," + d.parent.x
            + " " + d.parent.y + "," + d.parent.x;
    }

    svg.attr("transform", "scale(-1, 1) translate(0, 0)");
    svg
        .selectAll("text")
        .attr("transform", "scale(-1, 1) translate(-50, 0)")
        .style("text-anchor", "end")
        .attr("x", 20);
    svg.selectAll('text.dist-text').style('text-anchor', 'start')
    initZoom();
}


const updateCircleSizes = (n_clusters) => {
    nodes.selectAll('circle').attr("r", function (d) {
        return isFinalCluster(n_clusters, d.data.id) ? clusterCircleSize : nodeCircleSize
    })
}

const initZoom = () => {
    const svgImage = document.getElementById("svgImage");
    const svgContainer = document.getElementById("svgContainer");
    const zoomValue = document.getElementById("zoomValue");

    var viewBox = {x: 0, y: 0, w: svgImage.clientWidth, h: svgImage.clientHeight};
    svgImage.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);
    const svgSize = {w: svgImage.clientWidth, h: svgImage.clientHeight};
    var isPanning = false;
    var startPoint = {x: 0, y: 0};
    var endPoint = {x: 0, y: 0};
    var scale = 1;

    svgContainer.onmousewheel = function (e) {
        e.preventDefault();
        var w = viewBox.w;
        var h = viewBox.h;
        var mx = e.offsetX;//mouse x
        var my = e.offsetY;
        var dw = w * Math.sign(e.deltaY) * 0.05;
        var dh = h * Math.sign(e.deltaY) * 0.05;
        var dx = dw * mx / svgSize.w;
        var dy = dh * my / svgSize.h;
        viewBox = {x: viewBox.x + dx, y: viewBox.y + dy, w: viewBox.w - dw, h: viewBox.h - dh};
        scale = svgSize.w / viewBox.w;
        zoomValue.innerText = `${Math.round(scale * 100) / 100}`;
        svgImage.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);
    }
    svgContainer.onmousedown = function (e) {
        isPanning = true;
        startPoint = {x: e.x, y: e.y};
        dragged = false;
    }
    svgContainer.onmousemove = function (e) {
        if (isPanning) {
            endPoint = {x: e.x, y: e.y};
            var dx = (-startPoint.x + endPoint.x) / scale;
            var dy = (startPoint.y - endPoint.y) / scale;
            var movedViewBox = {x: viewBox.x + dx, y: viewBox.y + dy, w: viewBox.w, h: viewBox.h};
            svgImage.setAttribute('viewBox', `${movedViewBox.x} ${movedViewBox.y} ${movedViewBox.w} ${movedViewBox.h}`);
            dragged = true;
        }
    }

    svgContainer.onmouseup = function (e) {
        if (isPanning) {
            endPoint = {x: e.x, y: e.y};
            var dx = (-startPoint.x + endPoint.x) / scale;
            var dy = (startPoint.y - endPoint.y) / scale;
            viewBox = {x: viewBox.x + dx, y: viewBox.y + dy, w: viewBox.w, h: viewBox.h};
            svgImage.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);
            isPanning = false;
            setTimeout(() => {
                dragged = false;
            }, 100)
        }
    }
    svgContainer.onmouseleave = function (e) {
        isPanning = false;
    }

}

const getDendrogramData = () => {
    return fetch(
        '/experiment/' + experiment_id + '/dendrogram/' + word + '/data',
        {
            method: 'GET'
        }
    )
        .then((response) => response.json())
        .then((data) => {
            dataset = data;
            warmUpClusteringTree(dataset.tree)
            document.getElementById('n_clusters').setAttribute('max', Object.keys(data.data).length)
            return data
        });
}

const updateStyle = () => {
    let field = document.getElementById('color_field').value
    nodes.selectAll('text, circle').style(
        'fill', function (d) {
            return typeof d.data.label_id === 'undefined' && field === 'label_id' ? 'black' : colors[d.data[field]];
        }
    )
}

const setClusterDistances = (sourceNodeId) => {
    let allNodeTexts = d3.selectAll('text[class="dist-text"]')
    allNodeTexts.text('').style('font-weight', 'normal')

    if (sourceNodeId !== null) {
        let allNodes = getDescendants(dataset.tree.id)
        let descendantIds = getDescendants(sourceNodeId).map((n) => n.id)
        let distances = []

        allNodes.forEach((node) => {
            let nodeDescendants = getDescendants(node.id).map((n) => n.id)
            if (!descendantIds.includes(node.id) && !nodeDescendants.includes(sourceNodeId)) {
                if (sourceNodeId !== null && node.id !== sourceNodeId) {
                    distances.push([getDistanceBetweenNodes(sourceNodeId, node.id), node.id])
                }
            }
        })

        distances.sort()

        distances.forEach((item) => {
            let nodeId = item[1]
            let distance = item[0]
            let clusterNodeText = d3.selectAll('text[id="' + nodeId + '-dist"]')
            clusterNodeText.text(distance.toExponential(2)).style('font-size', '40px')
        })

        return distances
    }
}

const drawNearestNodesTable = (distances) => {
    let div = document.createElement('div');
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
    div.style.fontSize = '10px'
    let distanceElement = document.createElement('span')

    distanceElement.innerHTML = 'Nearest nodes (showing top 5 substitutes)'
    distanceElement.style.fontWeight = 'bold'

    div.append(distanceElement);

    distances.forEach((item) => {
        let neighbour = document.createElement('div')
        let dist = document.createElement('span')
        let topSubsts = document.createElement('span')
        topSubsts.style.textDecoration = 'underline'
        neighbour.style.cursor = 'pointer'
        neighbour.addEventListener('click', () => {
            selectNode(item[1])
        })

        let rankedSubsts = getRankedSubsts(getLeaves(item[1]))
        topSubsts.innerHTML = Object.keys(rankedSubsts).map((subst) => {
            return [rankedSubsts[subst], subst]
        }).sort((a, b) => b[0] - a[0]).slice(0, 5).map((s) => s[1]).join(', ')
        dist.innerHTML = ': ' + item[0].toExponential(2)
        neighbour.append(topSubsts)

        neighbour.append(dist)
        div.append(neighbour)
    })

    return div
}

const selectNode = (nodeId) => {
    toggleProcessingOverlay()
    setTimeout(() => {
        selectedNodeId = nodeId
        let leafNodes = []
        d3.selectAll("text").style('font-size', defaultFontSize)
        if (nodeId !== null) {
            leafNodes = getLeaves(nodeId)
            leafNodes.forEach((nId) => {
                d3.selectAll("text[id='" + nId + "']").style('font-size', selectedFontSize)
            })

        } else {
            setClusterDistances(null)
            summaryTableContainer.replaceChildren('')
        }

        if (leafNodes.length === 1) {
            summaryTableContainer.replaceChildren(drawSingleNodeSelectionSummaryTable(leafNodes[0]));
        } else {
            summaryTableContainer.replaceChildren(drawMultiNodeSelectionSummaryTable(leafNodes));
        }

        if (nodeId !== null) {
            let distances = setClusterDistances(nodeId)
            summaryTableContainer.append(drawNearestNodesTable(distances))
        }

        let trav = [];

        let q = [[dataset.tree, false]]

        while (q.length > 0) {
            let item = q.shift()
            let node = item[0]
            let found = item[1]
            trav.push(node.id)

            if (found) {
                links._groups[0][trav.indexOf(node.id) - 1].style.strokeWidth = '20px';
            } else {
                let link = links._groups[0][trav.indexOf(node.id) - 1]
                if (link) {
                    link.style.strokeWidth = '5px';
                }
            }
            if (node.id === nodeId) {
                found = true;
            }
            // second node is always the one with lower distance, thus reverse
            node.children.forEach((child) => {
                q.push([child, found])
            })
        }
        toggleProcessingOverlay()
    }, 0)
}

const drawClusterDistancesTable = (node1_id, node2_id) => {
    let div = document.createElement('div');
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
    div.style.fontSize = '10px'
    let distanceElement = document.createElement('span')

    distanceElement.innerHTML = '<u>Distance</u>: ' + getDistanceBetweenNodes(node1_id, node2_id).toExponential(2)
    distanceElement.innerHTML += ' <br /><u>JS divergence</u>: ' + getDivergence(node1_id, node2_id)

    div.append(distanceElement);

    let substs1 = getRankedSubsts(getLeaves(node1_id))
    let substs2 = getRankedSubsts(getLeaves(node2_id))

    let avgRanked = Object.keys(substs1).filter((s) => substs2[s]).map((subst) => {
        return [(substs1[subst] + substs2[subst]) / 2, subst, substs1[subst], substs2[subst]]
    }).sort((a, b) => a[0] - b[0])

    div.append(document.createElement('br'))

    avgRanked.forEach((subst) => {
        let substItem = document.createElement('b')
        substItem.innerText = subst[1] + ': '
        div.append(substItem)

        let rankItem = document.createElement('span')
        rankItem.innerText = 'Joint: ' + subst[0]
        div.append(rankItem)

        let rank1Item = document.createElement('span')
        rank1Item.innerText = ', 1st cluster: ' + subst[2]
        div.append(rank1Item)

        let rank2Item = document.createElement('span')
        rank2Item.innerText = ', 2nd cluster: ' + subst[3]
        div.append(rank2Item)

        div.append(document.createElement('br'))
    })


    return div;
}

const drawDiscriminativeSubsts = (node1_id, node2_id) => {
    let div = document.createElement('div');
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
    div.style.fontSize = '10px'
    let titleElement = document.createElement('span')

    titleElement.innerHTML = '<b>Discriminative substitutes</b>'
    div.append(titleElement)
    div.append(document.createElement('br'))

    let allSubstsWithProbs = getDiscriminativeSubsts(node1_id, node2_id)

    allSubstsWithProbs.slice(0, 10).forEach((item) => {
        div.append(drawDiscriminativeSubstsItem(item))
    })
    div.append(document.createElement('hr'))
    allSubstsWithProbs.slice(-10).forEach((item) => {
        div.append(drawDiscriminativeSubstsItem(item))
    })

    return div;
}