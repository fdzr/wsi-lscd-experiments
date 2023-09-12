let dataset = {};
let cache = {}
let idToNode = {}

const getEdgeId = (node1_id, node2_id) => {
    if (parseInt(node1_id) > parseInt(node2_id)) {
        return 'e_' + node1_id + '_' + node2_id
    }
    return 'e_' + node2_id + '_' + node1_id
}

const countSubsts = (nodeIds) => {
    let substs_counts = {}
    nodeIds.forEach((nodeId) => {
        let substs = [...new Set(dataset.data[nodeId]['substs'])];
        substs.forEach((subst) => {
            if (typeof substs_counts[subst] === 'undefined') {
                substs_counts[subst] = 0;
            }
            substs_counts[subst] += 1
        })
    });

    return substs_counts;
}

const getSubstsPandPMI = (nodeIds) => {
    let substs_pmi = []

    let substs_counts = countSubsts(nodeIds)
    let substs_counts_all = countSubsts(Object.keys(dataset.data))

    Object.keys(substs_counts).forEach((subst) => {
        let p = substs_counts[subst] / nodeIds.length
        substs_pmi.push([subst, p, Math.log2(p / (substs_counts_all[subst] / Object.keys(dataset.data).length))]);
    });

    return substs_pmi
}

const getRankedSubsts = (nodeIds) => {
    let substs_p_pmi = getSubstsPandPMI(nodeIds)
    substs_p_pmi.sort((a, b) => {
        return a[1] - b[1] || a[2] - b[2]
    }).reverse()

    let rank = 0
    let prev = [null, null, null];
    let substRank = {}

    substs_p_pmi.forEach((subst) => {
        if (subst[1] - prev[1] || subst[2] - prev[2]) {
            rank += 1
        }
        substRank[subst[0]] = rank
        prev = subst
    })
    return substRank
}

const toggleProcessingOverlay = () => {
    let overlay = document.getElementById('overlay')
    if (overlay) {
        overlay.remove()
        return
    }
    overlay = document.createElement('div')
    let text = document.createElement('div')
    overlay.setAttribute('id', 'overlay')
    overlay.style.position = 'fixed'
    overlay.style.top = '0'
    overlay.style.left = '0'
    overlay.style.width = '100%'
    overlay.style.height = '100%'
    overlay.style.opacity = '0.8'
    overlay.style.zIndex = '1000'
    overlay.style.backgroundColor = 'white'
    text.style.width = '100%'
    text.style.margin = 'auto 0'
    text.style.opacity = '1'
    text.style.position = 'absolute'
    text.style.top = '40%'
    text.style.textAlign = 'center'
    text.style.color = 'black'
    text.innerHTML = 'Processing... please wait.'
    overlay.append(text)
    document.body.append(overlay)
}

const getSubstsData = (nodeIds) => {

    let substs_p_pmi = getSubstsPandPMI(nodeIds)

    let substs_p = substs_p_pmi.slice().sort((a, b) => {
        return a[1] >= b[1] ? (a[1] === b[1] ? 0 : 1) : -1
    }).reverse()
    let substs_pmi = substs_p_pmi.slice().sort((a, b) => {
        return a[1] - b[1] || a[2] - b[2]
    }).reverse()

    let result = []

    const appendItem = (subst, div) => {
        let substItem = document.createElement('b')
        substItem.innerText = subst[0] + ': '
        div.append(substItem)

        let probItem = document.createElement('span')
        probItem.innerText = 'P: ' + (Math.round(subst[1] * 100) / 100)
        div.append(probItem)

        if (subst.length > 2) {
            let probItem = document.createElement('span')
            probItem.innerText = ', PMI: ' + (Math.round(subst[2] * 100) / 100)
            div.append(probItem)
        }

        div.append(document.createElement('br'))
    }

    let div = createSection('P(Sub | Cluster)')

    substs_p.forEach((subst) => {
        appendItem(subst, div)
    })

    result.push(div)

    div = createSection('P(Sub | Cluster), PMI(Sub, Cluster)')

    substs_pmi.forEach((subst) => {
        appendItem(subst, div)
    })

    result.push(div)
    return result;
}

const getMostFrequentSubsts = (nodeIds) => {
    let allSubsts = countSubsts(nodeIds)

    let allSubstsList = Object.keys(allSubsts).map((subst) => {
        return [allSubsts[subst], subst]
    })

    allSubstsList.sort().reverse()

    return allSubstsList
}

const createSection = (titleText) => {
    let div = document.createElement('div');
    div.style.width = '135px'
    div.style.overflow = 'hidden';
    div.style.float = 'left';
    div.style.padding = '5px';
    div.style.fontSize = '10px';
    div.style.text = '10px';
    div.className = 'substs-data-section'

    let title = document.createElement('div').innerText = titleText;
    div.append(title)
    div.append(document.createElement('br'))

    return div
}

const highlightedContext = (nodeId) => {
    let context = dataset.data[nodeId]['context']
    let parts = []

    parts.push(context.slice(0, dataset.data[nodeId]['begin']));
    parts.push(context.slice(dataset.data[nodeId]['begin'], dataset.data[nodeId]['end']))
    parts.push(context.slice(dataset.data[nodeId]['end']));

    context = parts[0] + '<b style="color:red">' + parts[1] + '</b>' + parts[2]

    return context
}

const drawSingleNodeSelectionSummaryTable = (nodeId) => {
    let context = highlightedContext(nodeId)
    let div = document.createElement('div');
    let contextDiv = document.createElement('div')
    let substs_probs = dataset.data[nodeId]['substs_probs']
    let substs = dataset.data[nodeId]['substs']
    let copyVector = document.createElement('a')

    let combined = substs.map((elem, i) => {
        return [substs_probs[i], elem]
    })

    copyVector.addEventListener('click', () => {
        navigator.clipboard.writeText('[' + dataset.vectors[nodeId] + ']')
    })

    copyVector.innerHTML = 'Copy vector'
    copyVector.style.position = 'absolute'
    copyVector.style.cursor = 'pointer'
    copyVector.style.top = '5px'
    copyVector.style.right = '5px'

    combined.sort().reverse()
    contextDiv.innerHTML = '<u>Context</u>: ' + context + '<br />';

    div.append(contextDiv)

    let neighbouringSubstsSection = createSection('Substs by nearby freq.')

    let neighbours = getNearestNeighbours(nodeId, 10)

    let mostFrequentSubstsAmongNeighbours = getMostFrequentSubsts(neighbours.map((n) => n[1]))

    let mostFrequentSubstsOverlappingWithNeighbours = mostFrequentSubstsAmongNeighbours.filter(subst => substs.includes(subst[1]))

    mostFrequentSubstsOverlappingWithNeighbours.forEach((substFreqPair) => {
        let subst = substFreqPair[1]
        let freq = substFreqPair[0]
        let b = document.createElement('b')
        let span = document.createElement('span')
        b.innerHTML = subst
        neighbouringSubstsSection.append(b)
        neighbouringSubstsSection.append(': ')
        span.innerHTML = freq
        neighbouringSubstsSection.append(span)
        neighbouringSubstsSection.append(document.createElement('br'))
    })

    let substsSection = createSection('Substitutes by probability')

    combined.map((substWithProb) => {
        let b = document.createElement('b')
        let span = document.createElement('span')
        b.innerHTML = substWithProb[1]
        substsSection.append(b)
        substsSection.append(': ')
        span.innerHTML = substWithProb[0].toExponential(2)
        substsSection.append(span)
        substsSection.append(document.createElement('br'))
    })

    div.append(substsSection)
    div.append(neighbouringSubstsSection)
    div.append(copyVector)

    div.style.position = 'relative';
    div.style.border = 'solid 1px #cacaca';
    div.style.background = '#fff';
    div.style.padding = '5px';
    div.style.paddingTop = '25px';
    div.style.width = 'calc(100% - 10px)';
    div.style.height = '350px';
    div.style.borderRadius = '3px';
    div.style.overflow = 'scroll';
    div.style.zIndex = '1000';
    div.style.marginBottom = '10px';
    return div;
}

const drawMultiNodeSelectionSummaryTable = (nodeIds) => {
    if (nodeIds.length === 0) {
        return '';
    }
    let div = document.createElement('div');
    div.style.position = 'relative';
    div.style.border = 'solid 1px #cacaca';
    div.style.background = '#fff';
    div.style.padding = '5px';
    div.style.paddingTop = '25px';
    div.style.width = 'calc(100% - 10px)';
    div.style.height = '350px';
    div.style.borderRadius = '3px';
    div.style.overflow = 'scroll';
    div.style.zIndex = '1000';
    div.style.marginBottom = '10px';
    div.className = 'multi-node-summary-container'

    getSubstsData(nodeIds).forEach((section) => {
        div.append(section)
    });
    return div;
}


const createToolTip = (nodeId, x, y) => {
    let context = highlightedContext(nodeId)

    let div = document.createElement('div');
    div.innerHTML = '<u>Context</u>: ' + context;
    div.innerHTML += '<br /><br />(Click to select the node and see more info)'
    div.style.position = 'absolute';
    div.style.top = y + 'px';
    div.style.left = x + 'px';
    div.style.border = 'solid 1px #cacaca';
    div.style.background = '#fff';
    div.style.padding = '5px';
    div.style.maxWidth = '250px';
    div.style.borderRadius = '3px';
    div.id = 'tooltip';
    return div;
}

const getNearestNeighbours = (nodeId, K) => {
    let neighbours = []

    Object.keys(dataset.data).forEach((id) => {
        if (id !== nodeId) {
            let edgeId = getEdgeId(id, nodeId)
            neighbours.push([
                dataset.distances[edgeId],
                id
            ])
        }
    });
    neighbours.sort()
    return neighbours.slice(0, K)
}

const isFinalCluster = (n_clusters, nodeId) => {
    return getFinalClusterNodes(n_clusters).includes(nodeId)
}

const getClusterStateTraversal = (n, nodeId) => {
    let q = [getNodeById(nodeId)]
    let result = []

    while (q.length > 0 && n > 1) {

        let node = q.shift()
        result.push(node)
        n--;
        node.children.forEach((child) => {
            if (child.children.length === 0) {
                child.dist = -1
            }
            q.push(child)
        })
        q.sort((a, b) => a.dist < b.dist ? 1 : -1)

    }

    return result.concat(q)
}

const getFinalClusterNodes = (n_clusters) => {
    return getClusterStateTraversal(n_clusters, dataset.tree.id).slice(-n_clusters).map((n) => n.id)
}

const getClusterNodes = () => {
    let cacheKey = 'getClusterNodes'
    if (typeof cache[cacheKey] !== 'undefined') {
        return cache[cacheKey]
    }
    cache[cacheKey] = getClusterStateTraversal(Object.keys(dataset.data).length * 2 - 1, dataset.tree.id).map((n) => n.id)
    return cache[cacheKey]
}

const warmUpClusteringTree = () => {
    let q = [dataset.tree]
    while (q.length > 0) {
        let node = q.shift()
        node['id'] = parseInt(node['id'])
        cache['node_' + node['id']] = node
        node.children.forEach((child) => {
            child.parentNodeId = node.id
            q.push(child)
        })
    }
}

const getNodeById = (nodeId) => {
    let cacheKey = 'node_' + nodeId
    if (typeof cache[cacheKey] !== 'undefined') {
        return cache[cacheKey]
    }

    let q = [dataset.tree]

    while (q.length > 0) {
        let node = q.shift()
        if (nodeId === node.id) {
            cache[cacheKey] = node
            return node
        }
        node.children.forEach((child) => {
            q.push(child)
        })
    }

}

const getLeaves = (nodeId) => {

    let q = [getNodeById(nodeId)]
    let leaves = []

    while (q.length > 0) {
        let node = q.shift()

        if (node.children.length === 0) {
            leaves.push(node.id)
        }
        node.children.forEach((child) => {
            q.push(child)
        })
    }
    return leaves
}

const getDescendants = (nodeId) => {
    let q = [getNodeById(nodeId)]
    let result = []

    while (q.length > 0) {

        let node = q.shift()
        result.push(node)
        node.children.forEach((child) => {

            q.push(child)
        })

    }
    return result

}

const getDivergence = (node1_id, node2_id) => {

    let cacheKey = 'divergence_' + getEdgeId(node1_id, node2_id)

    if (typeof cache[cacheKey] !== 'undefined') {
        return cache[cacheKey]
    }

    let leaves1 = getLeaves(node1_id)
    let leaves2 = getLeaves(node2_id)

    let counts1 = countSubsts(leaves1)
    let counts2 = countSubsts(leaves2)
    let allCounts = countSubsts(getLeaves(dataset.tree.id))

    Object.keys(allCounts).forEach((subst) => {
        if (typeof counts1[subst] === 'undefined') {
            counts1[subst] = 0
        }
        if (typeof counts2[subst] === 'undefined') {
            counts2[subst] = 0
        }
    })

    let expSumCounts1 = Object.keys(counts1).reduce((agg, e) => Math.exp(Math.log(10) * counts1[e]) + agg, 0)
    let probs1 = Object.keys(counts1).map((subst) => [subst, Math.exp(Math.log(10) * counts1[subst]) / expSumCounts1])
    probs1.sort((a, b) => a[0] > b[0] ? 1 : -1)
    probs1 = probs1.map((s) => s[1])

    let expSumCounts2 = Object.keys(counts2).reduce((agg, e) => Math.exp(Math.log(10) * counts2[e]) + agg, 0)
    let probs2 = Object.keys(counts2).map((subst) => [subst, Math.exp(Math.log(10) * counts2[subst]) / expSumCounts2])
    probs2.sort((a, b) => a[0] > b[0] ? 1 : -1)
    probs2 = probs2.map((s) => s[1])

    let probsM = probs1.map((e, i) => (e + probs2[i]) / 2)

    let kl1M = probs1.reduce((agg, e, i) => agg + e * (Math.log2(e) - Math.log2(probsM[i])), 0)
    let kl2M = probs2.reduce((agg, e, i) => agg + e * (Math.log2(e) - Math.log2(probsM[i])), 0)

    cache[cacheKey] = (kl1M + kl2M) / 2
    return cache[cacheKey]
}

const getDistanceBetweenNodes = (node1_id, node2_id) => {

    let cacheKey = 'distance_' + getEdgeId(node1_id, node2_id)

    if (typeof cache[cacheKey] !== 'undefined') {
        return cache[cacheKey]
    }

    let leaves1 = getLeaves(node1_id)
    let leaves2 = getLeaves(node2_id)

    let total = 0
    let count = 0

    leaves1.forEach((leaf1_id) => {
        leaves2.forEach((leaf2_id) => {
            let edgeId = getEdgeId(leaf1_id, leaf2_id)

            total += dataset.distances[edgeId]
            count += 1

        })
    })

    cache[cacheKey] = (total / count)
    return cache[cacheKey]
}

const drawSubsts = (nodeIds, n) => {
    nodeIds.forEach((nodeId) => {
        let node = dataset.data[nodeId]
        let container = document.getElementById('substs-' + nodeId)
        if (container) {
            let sorted_substs = node.substs.map((_, i) => [node.substs_probs[i], node.substs[i]]).sort((a, b) => b[0] - a[0]).slice(0, n)
            if (n > 0) {
                container.style.display = 'block'
                container.innerHTML = '<td class="subs mdl-data-table__cell--non-numeric">' + sorted_substs.map((s) => '<b>' + s[1] + '</b>: ' + s[0].toExponential(2)).join(', ') + '</td>'
            } else {
                container.style.display = 'none'
                container.innerHTML = '';
            }
        }
    })
}

const drawSamplesList = (samples, corpus) => {

    let nSubsts = document.createElement('input')
    nSubsts.setAttribute('type', 'number')
    nSubsts.setAttribute('value', 0)
    nSubsts.setAttribute('min', 0)
    nSubsts.style.marginBottom = '5px'
    nSubsts.addEventListener('change', (e) => {
        drawSubsts(samples.map(s => s.id), e.target.value)
    })

    let corpora = document.createElement('select')
    corpora.innerHTML = '<option value="-1" ' + (corpus === -1 ? 'selected' : '') + '>All corpora</option>'
    corpora.innerHTML += '<option value="0" ' + (corpus === 0 ? 'selected' : '') + '>' + dataset.corpora_names[0] + '</option>'
    corpora.innerHTML += '<option value="1" ' + (corpus === 1 ? 'selected' : '') + '>' + dataset.corpora_names[1] + '</option>'

    let container = document.createElement('div')
    samples.filter(s => corpus === -1 || s.corpus === corpus).forEach((node) => {
        let sample = document.createElement('table')
        let context = highlightedContext(node.id)
        sample.className = 'examples-table overview-table mdl-data-table mdl-js-data-table mdl-shadow--0dp'
        sample.innerHTML = '<tbody>' +
            '<tr>' +
            '<td class="context mdl-data-table__cell--non-numeric">' + context + '</td>' +
            '</tr>' +
            '<tr id="substs-' + node.id + '" style="height:auto"></tr>' +
            '</tbody>'
        container.append(sample)
        container.append(document.createElement('br'))
    })
    return {
        samplesContainer: container,
        nSubstsControl: nSubsts,
        corporaControl: corpora
    };
}

const nClustersSliderChange = (target, callback) => {
    n_clusters = document.getElementById(target).value

    let text = document.getElementById(target + '-label').innerHTML

    let parts = text.split(',')
    let nClustersText = parts[0]
    let nIterationsText = parts[1]

    let updatesTexts = []
    updatesTexts.push(nClustersText.split(' ').slice(0, -1).join(' ') + ' ' + n_clusters)
    updatesTexts.push(nIterationsText.split(' ').slice(0, -1).join(' ') + ' ' + (Object.keys(dataset.data).length - n_clusters))

    document.getElementById(target + '-label').innerHTML = updatesTexts.join(', ')

    callback(n_clusters)
}

const getDiscriminativeSubsts = (node1_id, node2_id) => {
    let counts1 = cache['subst_counts_' + node1_id] || countSubsts(getLeaves(node1_id))
    cache['subst_counts_' + node1_id] = counts1
    let counts2 = cache['subst_counts_' + node2_id] || countSubsts(getLeaves(node2_id))
    cache['subst_counts_' + node2_id] = counts2
    let vocabSize = Object.keys(counts1).length + Object.keys(counts2).length

    let allSubsts = [...new Set(Object.keys(counts1).concat(Object.keys(counts2)))]

    let allSubstsWithProbs = []

    allSubsts.forEach((subst) => {
        let countSense1 = counts1[subst] ? counts1[subst] : 0
        let countSense2 = counts2[subst] ? counts2[subst] : 0
        let prob1 = (countSense1 + 1) / (countSense1 + vocabSize)
        let prob2 = (countSense2 + 1) / (countSense1 + vocabSize)
        allSubstsWithProbs.push([prob1 / prob2, subst, prob1, prob2])
    })

    allSubstsWithProbs.sort().reverse()

    return allSubstsWithProbs
}

const drawDiscriminativeSubstsItem = (item) => {
    let div = document.createElement('div')
    let subst = item[1]
    let prob1 = item[2]
    let prob2 = item[3]

    let substItem = document.createElement('b')
    substItem.innerText = subst
    div.append(substItem)

    let prob1Elem = document.createElement('span')
    prob1Elem.innerText = ' ' + prob1.toExponential(3)
    div.append(prob1Elem)

    let prob2Elem = document.createElement('span')
    prob2Elem.innerText = ' / ' + prob2.toExponential(3)
    div.append(prob2Elem)

    return div;
}

const getCountsHistogramConfig = () => {
    let labels = dataset.counts[0].data.map((_, i) => i);

    return {
        type: 'bar',
        data: {
            labels: labels,
            datasets: dataset.counts
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Number of items per cluster per corpus'
                }
            }
        },
    };
}

const getSilARIChartConfig = () => {
    let data = {
        labels: dataset.n_clusters_list,
        datasets: [
            {
                label: 'Silhouette score',
                data: dataset.sil_scores,
                borderColor: 'red',
                yAxisID: 'y',
            }
        ]
    }
    if (dataset.ari_scores.length > 0) {
        data.datasets.push({
            label: 'ARI score',
            data: dataset.ari_scores,
            borderColor: 'blue',
            yAxisID: 'y',
        })
    }
    return {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            stacked: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Silhouette score' + (dataset.ari_scores.length > 0 ? ' and ARI score' : '') + ' per cluster number'
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',

                    // grid line settings
                    grid: {
                        drawOnChartArea: false, // only want the grid lines for one axis to show up
                    },
                },
            }
        },
    }
}


const getVectorDistancesConfig = () => {
    let labels = dataset.distances[0].x.map((x) => x.toExponential(2));

    return {
        type: 'bar',
        data: {
            labels: labels,
            datasets: dataset.distances
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Intra and inter corpus vector distances'
                }
            }
        },
    };
}