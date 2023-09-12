const colors = ['orange', 'blue', 'red', 'black', 'purple', 'green']
const cont = document.getElementById('clusters-container')
let clusterCountsHistogram = null;
let vectorDistancesHistogram = null;
let silARILineChart = null;
let n_clusters = num_clusters

document.addEventListener('DOMContentLoaded', () => {
    toggleProcessingOverlay()
    getClusteringData().then(() => {
        prepare(dataset.tree)
        plotCharts()
        draw(false)
        refreshSenseFilter()
        toggleProcessingOverlay()
    });
});

const draw = (onlyParents) => {
    drawClusters(
        filter((node) =>
            getFilter('tags')(node) &&
            getFilter('sense')(node) &&
            getFilter('min_samples')(node) &&
            getFilter('max_samples')(node),
        onlyParents).map(n => n.id)
    )
}

const getFilter = (type) => {
    if (type === 'sense') {
        let sense = document.getElementById('filter-senses').value

        return node => dataset.senses[node.id] === sense ||
            (sense === 'none' && typeof dataset.senses[node.id] === 'undefined') ||
            (sense === 'any')

    } else if (type === 'tags') {
        let tag = document.getElementById('filter-tag').value
        switch (tag) {
            case 'any':
                return node => true
            case 'cluster':
                return node => isFinalCluster(n_clusters, node.id)
            case 'increased_entropy':
                return node => node.increasedEntropy
            case 'high_pmi':
                return node => node.highPMI
            case 'imbalanced_merge':
                return node => node.isImbalancedMerge
        }
    } else if (type === 'min_samples') {
        let minSamples = document.getElementById('filter-min-samples').value
        return node => node.countLeaves >= minSamples
    } else if (type === 'max_samples') {
        let maxSamples = document.getElementById('filter-max-samples').value
        return node => node.countLeaves <= maxSamples
    }
}

const drawClusters = (clusterNodeIds) => {
    cont.innerHTML = ''

    let headers = document.createElement('div')

    let headerChildren = document.createElement('div')
    headerChildren.classList.add('cluster-header')
    headerChildren.innerHTML = 'Child clusters'
    headers.append(headerChildren)

    let headerCluster = document.createElement('div')
    headerCluster.classList.add('cluster-header')
    headerCluster.innerHTML = 'Cluster'
    headers.append(headerCluster)

    let headerParent = document.createElement('div')
    headerParent.classList.add('cluster-header')
    headerParent.innerHTML = 'Parent cluster'
    headers.append(headerParent)

    cont.append(headers)

    clusterNodeIds.map(nodeId => getNodeById(nodeId)).sort((a, b) => a.parentNodeId - b.parentNodeId).map(node => node.id).forEach(clusterNodeId => {
        cont.append(drawClusterBranch(clusterNodeId))
    })

    refreshSenseStats()
}

const drawClusterBranch = (clusterNodeId) => {
    let clusteringNode = getNodeById(clusterNodeId)
    let branch = document.createElement('div')
    let existingClusterBlock = false;
    let emptyChildren = null;
    branch.style.opacity = '0'
    branch.setAttribute('class', 'branch')

    let children = document.createElement('div')
    children.classList.add('cluster-children')

    if (clusteringNode.children.length > 0) {
        let discriminativeSubsts = ''; // getDiscriminativeSubsts(clusteringNode.children[0].id, clusteringNode.children[1].id)
                                       // not included due ro performance reasons
                                       // TODO: this function to onclick callback for lazy loading


        children.append(drawCluster(clusteringNode.children[0].id, discriminativeSubsts.slice(0, 10)))
        children.append(drawCluster(clusteringNode.children[1].id, discriminativeSubsts.slice(-10)))

    } else {
        emptyChildren = document.createElement('div')
        emptyChildren.classList.add('empty-children', 'cluster-block')
        emptyChildren.innerHTML = 'Leaf node, no children'
        children.append(emptyChildren)
    }

    branch.append(children)

    let cluster = document.createElement('div')

    cluster.setAttribute('class', 'cluster-cluster')
    cluster.append(drawCluster(clusterNodeId))
    branch.append(cluster)

    let parent = document.createElement('div')

    parent.setAttribute('class', 'cluster-parent')

    if (clusteringNode.parentNodeId) {
        existingClusterBlock = document.getElementById('cluster-block-' + clusteringNode.parentNodeId)

        if (!existingClusterBlock) {
            parent.append(drawCluster(clusteringNode.parentNodeId))
        }
    }

    branch.append(parent)

    setTimeout(() => {
        if (!emptyChildren) {
            cluster.style.height = (children.clientHeight - 20) + 'px'
        } else {
            children.style.height = (cluster.clientHeight - 20) + 'px'
            cluster.classList.add('has-no-children')
        }

        if (existingClusterBlock) {
            existingClusterBlock.parentElement.classList.add('parent-of-two')
        } else {
            parent.style.height = (cluster.clientHeight - (emptyChildren ? 20 : 0)) + 'px'
        }

        branch.style.opacity = '1'
    }, 0)

    return branch
}

const filter = (callback, onlyParents) => {
    let q = [dataset.tree]
    let result = []
    let maxResults = document.getElementById('filter-max-results').value

    while (q.length > 0) {
        let node = q.shift()
        let include = callback(node)

        if (include) {
            result.push(node)
        }

        if (result.length >= maxResults) {
            return result
        }

        if (!include || !onlyParents) {
            node.children.forEach((child) => {
                q.push(child)
            })
        }
    }

    return result
}

const prepare = (node) => {
    node.countLabels = {}
    node.corpus = dataset.data[node.id] ? dataset.data[node.id].corpus : null
    if (node.label_id !== null) {
        node.countLabels[node.label_id] = 1
    }
    node.countCorpora = {}
    if (node.corpus !== null) {
        node.countCorpora[node.corpus] = 1
    }
    node.countLeaves = 0
    node.isImbalancedMerge = false
    node.increasedEntropy = false
    node.iteration = getIteration(node.id)
    node.iteration = node.iteration > 0 ? node.iteration : '0 (sample)'
    node.leaves = []
    node.entropy = 0
    if (node.children.length > 0) {
        let leavesNums = []
        node.children.forEach((child) => {
            prepare(child)
            Object.keys(child.countLabels).forEach((i) => node.countLabels[i] = (node.countLabels[i] || 0) + child.countLabels[i])
            Object.keys(child.countCorpora).forEach((i) => node.countCorpora[i] = (node.countCorpora[i] || 0) + child.countCorpora[i])
            node.countLeaves += child.countLeaves
            leavesNums.push(child.countLeaves)
            node.leaves = node.leaves.concat(child.leaves)
        })
        node.isImbalancedMerge = Math.max(...leavesNums) * 0.2 >= Math.min(...leavesNums);

        node.entropy = getEntropy(node.countLabels, node.countLeaves)

        if (node.entropy > node.children.reduce((a, b) => a.entropy + b.entropy) / node.children.length) {
            node.increasedEntropy = true
        }
    } else {
        node.leaves = [node.id]
        node.countLeaves = 1

    }
    node.pmis = [
        (node.countCorpora[0] > 0) ? getPMI(node.countCorpora[0] / node.countLeaves, 0) : -99,
        (node.countCorpora[1] > 0) ? getPMI(node.countCorpora[1] / node.countLeaves, 1) : -99,
    ]

    if (typeof node.countCorpora[1] !== 'undefined' && (Math.abs(node.pmis[0]) >= 2 || Math.abs(node.pmis[1]) >= 2)) {
        node.highPMI = true
    }

    return node
}

const getEntropy = (countLabels, countLeaves) => {
    let entropy = 0

    Object.keys(countLabels).forEach((label) => entropy -= (countLabels[label] / countLeaves) * Math.log2(countLabels[label] / countLeaves))

    return entropy
}

const getPMI = (pCorpusInCluster, corporaNum) => {
    return Math.log2(pCorpusInCluster) - Math.log2(getNumCorpus(corporaNum) / Object.keys(dataset.data).length)
}

const getNumCorpus = (corpus) => {

    let cacheKey = 'nCorpus_' + corpus
    if (typeof cache[cacheKey] !== 'undefined') {
        return cache[cacheKey]
    }

    cache[cacheKey] = Object.keys(dataset.data).filter(i => dataset.data[i].corpus === corpus).length
    return cache[cacheKey]
}

const getSenses = () => {
    return Object.values(dataset.senses).filter((value, index, self) => self.indexOf(value) === index && value !== 'Multiple senses')
}

const setSense = (nodeId, sense) => {

    if (sense === '') {
        return
    }

    let q = [getNodeById(nodeId)]

    while (q.length > 0) {
        let node = q.shift()

        dataset.senses[node.id] = sense

        refreshSenses(node)

        node.children.forEach(child => {
            q.push(child)
        })
    }

    let parent = getNodeById(getNodeById(nodeId).parentNodeId)

    while (parent) {
        let child1sense = dataset.senses[parent.children[0].id]
        let child2sense = dataset.senses[parent.children[1].id]

        if (child1sense === child2sense && child1sense) {
            dataset.senses[parent.id] = child1sense
        } else if (child1sense && child2sense) {
            dataset.senses[parent.id] = 'Multiple senses'

        } else {
            delete dataset.senses[parent.id]
        }

        refreshSenses(parent)

        parent = getNodeById(parent.parentNodeId)
    }

    refreshSenseStats()
    refreshSenseFilter()
}

const hasSense = (nodeId) => {
    return typeof dataset.senses[nodeId] !== 'undefined' && dataset.senses[nodeId] !== 'Multiple senses'
}

const refreshSenseStats = () => {
    let allSenses = getSenses()
    let sensesContainer = document.getElementById('senses-container')
    sensesContainer.innerHTML = ''

    let totals = document.createElement('h5')
    totals.innerHTML = '<b>Samples without a sense:</b> ' + (Object.keys(dataset.data).filter(nodeId => !hasSense(nodeId)).length)
    sensesContainer.append(totals)

    allSenses.forEach(sense => {
        let stats = getSenseStats(sense)
        sensesContainer.append(drawSenseStats(stats))
    })
    if (allSenses.length > 0) {
        let saveSensesButton = document.createElement('button')
        saveSensesButton.innerHTML = 'Save'
        saveSensesButton.onclick = () => {
            saveSensesButton.innerHTML = 'Saving...'
            saveSenses().then(() => {
                saveSensesButton.innerHTML = 'Saved'
                setTimeout(() => {
                    saveSensesButton.innerHTML = 'Save'
                }, 3000)
            })
        }
        sensesContainer.append(saveSensesButton)
    }

}

const refreshSenseFilter = () => {
    let senseFilterEle = document.getElementById('filter-senses')
    senseFilterEle.innerHTML = ['any', 'none'].concat(
        getSenses()).map(sense => '<option value="' + sense + '">' + sense + '</option>').join('')
}

const drawSenseStats = (stats) => {
    let senseBlock = document.createElement('div')

    senseBlock.classList.add('sense-block')

    let senseName = document.createElement('div')
    senseName.innerHTML = '<b>Sense name: </b>' + stats.sense
    senseBlock.append(senseName)

    let senseSignature = document.createElement('div')
    senseSignature.innerHTML = '<b>Signature: </b>' + stats.signature
    senseBlock.append(senseSignature)

    let totalSamples = document.createElement('div')
    totalSamples.innerHTML = '<b>Total Samples:</b> ' + stats.counts.totalSamples
    senseBlock.append(totalSamples)

    if (Object.keys(dataset.tree.countLabels).length > 1) {
        let labelCounts = document.createElement('div')
        labelCounts.innerHTML = '<b>Class counts: </b>'
        let labelCountsList = []
        let nonZeroLabel = false;
        Object.keys(stats.counts.labels).forEach((label) => {
            labelCountsList.push(label + ': ' + stats.counts.labels[label])
            if (stats.counts.labels[label] > 0) {
                nonZeroLabel = true
            }
        })
        if (nonZeroLabel) {
            labelCounts.innerHTML += labelCountsList.join(', ')
            senseBlock.append(labelCounts)
        }

        let entropy = document.createElement('div')
        entropy.innerHTML = '<b>Entropy:</b> ' + stats.entropy
        senseBlock.append(entropy)
    }

    if (Object.keys(dataset.corpora_names).length > 1) {
        let corpusCounts = document.createElement('div')
        corpusCounts.innerHTML = '<b>Corpora: </b>'
        let corpusCountsList = []
        let nonZeroCorpus = false;
        Object.keys(stats.counts.corpora).forEach((corpus) => {
            corpusCountsList.push(dataset.corpora_names[corpus] + ': ' + stats.counts.corpora[corpus])
            if (stats.counts.corpora[corpus] > 0) {
                nonZeroCorpus = true
            }
        })
        if (nonZeroCorpus) {
            corpusCounts.innerHTML += corpusCountsList.join(', ')
            senseBlock.append(corpusCounts)
        }

        let pmi = document.createElement('div')
        let pmi0 = stats.pmis[0] === '-Inf' ? -99 : stats.pmis[0]
        let pmi1 = stats.pmis[1] === '-Inf' ? -99 : stats.pmis[1]

        pmi.innerHTML = '<b>PMI(' + dataset.corpora_names[0] + ', Sense):</b> ' + stats.pmis[0] + ', <b>PMI(' + dataset.corpora_names[1] + ', Sense):</b>' + stats.pmis[1]
        senseBlock.append(pmi)

        let gradient = document.createElement('div')
        gradient.style.height = '20px'
        gradient.style.width = '300px'
        gradient.style.margin = '3px 0'

        let left = Math.pow(1.5, pmi0)
        let right = Math.pow(1.5, pmi1)

        let splitLeft = left / (left + right)

        let leftBar = document.createElement('div')
        let rightBar = document.createElement('div')

        leftBar.style.background = 'red'
        leftBar.style.width = 100 * splitLeft + '%'
        leftBar.style.height = '20px'
        leftBar.style.display = 'inline-block'

        rightBar.style.background = 'blue'
        rightBar.style.width = (100 - 100 * splitLeft) + '%'
        rightBar.style.height = '20px'
        rightBar.style.display = 'inline-block'

        gradient.append(leftBar)
        gradient.append(rightBar)
        senseBlock.append(gradient)
    }

    return senseBlock
}

const getSenseStats = (sense) => {
    let leafNodeIds = Object.keys(dataset.senses).filter((nodeId) => dataset.senses[nodeId] === sense && getNodeById(nodeId).children.length === 0)
    let counts = getSenseCounts(leafNodeIds)
    let signature = getClusterSignature(leafNodeIds)
    let pmis = [
        (counts.corpora[0] > 0) ? getPMI(counts.corpora[0] / counts.totalSamples, 0) : '-Inf',
        (counts.corpora[1] > 0) ? getPMI(counts.corpora[1] / counts.totalSamples, 1) : '-Inf',
    ]

    let entropy = getEntropy(counts.labels, counts.totalSamples)

    return {
        counts: counts,
        signature: signature,
        pmis: pmis,
        entropy: entropy,
        sense: sense
    }
}

const getSenseCounts = (leafNodeIds) => {
    let counts = {
        corpora: {},
        labels: {},
        totalSamples: 0
    }

    leafNodeIds.forEach(nodeId => {
        let node = getNodeById(nodeId)
        if (typeof counts.corpora[node.corpus] === 'undefined') {
            counts.corpora[node.corpus] = 0
        }
        if (typeof counts.labels[node.label_id] === 'undefined') {
            counts.labels[node.label_id] = 0
        }
        counts.labels[node.label_id] += 1
        counts.corpora[node.corpus] += 1
        counts.totalSamples += 1
    })

    return counts

}

const drawCluster = (clusterNodeId, discriminativeSubsts) => {

    let samplesOpen = false;
    let discrSubstsOpen = false;
    let node = getNodeById(clusterNodeId)
    let isFinal = isFinalCluster(n_clusters, clusterNodeId)
    let drawSamplesButton = document.createElement('a')
    let drawDiscriminativeSubstsButton = document.createElement('a')
    let setSenseButton = document.createElement('button')
    let samplesContainer = document.createElement('div')
    let discriminativeSubstsContainer = document.createElement('div')
    let clusterBlock = document.createElement('div')
    let sensesList = document.createElement('select')
    let newSense = document.createElement('input')

    newSense.setAttribute('type', 'text')
    newSense.setAttribute('placeholder', '... or add a new sense')

    newSense.setAttribute('id', 'new-sense-' + node.id)

    sensesList.setAttribute('id', 'senses-' + node.id)
    sensesList.classList.add('senses-list')

    clusterBlock.setAttribute('id', 'cluster-block-' + clusterNodeId)

    drawSamplesButton.innerHTML = 'Show samples'
    drawSamplesButton.style.cursor = 'pointer'

    drawDiscriminativeSubstsButton.innerHTML = 'Show discriminative substitutes'
    drawDiscriminativeSubstsButton.style.cursor = 'pointer'

    setSenseButton.innerHTML = 'Set sense'
    setSenseButton.style.cursor = 'pointer'

    samplesContainer.setAttribute('class', 'samples-list')
    discriminativeSubstsContainer.setAttribute('class', 'discriminative-substs-list')

    let tags = document.createElement('div')
    tags.setAttribute('class', 'tags')
    tags.setAttribute('id', 'tags-' + node.id)

    let signature = document.createElement('div')
    signature.setAttribute('class', 'signature')
    node.signature = node.signature || getClusterSignature(node.leaves)
    signature.innerHTML = node.signature

    if (node.countLeaves > 1 && node.countLeaves <= Object.keys(dataset.data).length - 1) {
        signature.style.cursor = 'pointer'
        signature.onclick = () => {
            drawClusters([clusterNodeId])
        }
    }
    clusterBlock.append(signature)

    let id = document.createElement('div')
    id.innerHTML = '<b>ID: </b>' + clusterNodeId
    clusterBlock.append(id)

    let numLeaves = document.createElement('div')
    numLeaves.innerHTML = '<b>Total samples: </b>' + node.countLeaves
    clusterBlock.append(numLeaves)

    let iteration = document.createElement('div')
    iteration.innerHTML = '<b>Iteration: </b>' + node.iteration
    clusterBlock.append(iteration)

    if (Object.keys(dataset.tree.countLabels).length > 1) {
        let labelCounts = document.createElement('div')
        labelCounts.innerHTML = '<b>Class counts: </b>'
        let labelCountsList = []
        let nonZeroLabel = false;
        Object.keys(node.countLabels).forEach((label) => {
            labelCountsList.push(label + ': ' + node.countLabels[label])
            if (node.countLabels[label] > 0) {
                nonZeroLabel = true
            }
        })
        if (nonZeroLabel) {
            labelCounts.innerHTML += labelCountsList.join(', ')
            clusterBlock.append(labelCounts)
        }

        let entropy = document.createElement('div')
        entropy.innerHTML = '<b>Entropy: </b>' + node.entropy
        clusterBlock.append(entropy)
    }

    if (Object.keys(dataset.corpora_names).length > 1) {
        let corpusCounts = document.createElement('div')
        corpusCounts.innerHTML = '<b>Corpora: </b>'
        let corpusCountsList = []
        let nonZeroCorpus = false;
        Object.keys(node.countCorpora).forEach((corpus) => {
            corpusCountsList.push(dataset.corpora_names[corpus] + ': ' + node.countCorpora[corpus])
            if (node.countCorpora[corpus] > 0) {
                nonZeroCorpus = true
            }
        })
        if (nonZeroCorpus) {
            corpusCounts.innerHTML += corpusCountsList.join(', ')
            clusterBlock.append(corpusCounts)
        }

        let pmi = document.createElement('div')
        pmi.innerHTML = '<b>PMI(' + dataset.corpora_names[0] + ', Cluster): </b>' + node.pmis[0] + '<b>, PMI(' + dataset.corpora_names[1] + ', Cluster): </b>' + node.pmis[1]

        clusterBlock.append(pmi)
    }

    clusterBlock.append(tags)
    clusterBlock.append(drawSamplesButton)

    if (discriminativeSubsts) {
        clusterBlock.append(drawDiscriminativeSubstsButton)
    }

    clusterBlock.append(sensesList)
    clusterBlock.append(newSense)
    clusterBlock.append(setSenseButton)

    clusterBlock.append(samplesContainer)
    clusterBlock.append(discriminativeSubstsContainer)

    drawSamplesButton.onclick = () => {
        if (samplesOpen) {
            samplesContainer.innerHTML = ''
            drawSamplesButton.innerHTML = 'Show samples'
            samplesOpen = false;
            clusterBlock.parentElement.classList.remove('samples-opened')
            return
        }

        let nodeSamples = node.leaves.map(nodeId => getNodeById(nodeId))
        let samplesElements = drawSamplesList(nodeSamples, -1)

        const refreshList = (samplesElements) => {
            samplesContainer.innerHTML = ''
            samplesContainer.append(samplesElements.nSubstsControl)
            samplesContainer.append(samplesElements.corporaControl)
            samplesContainer.append(samplesElements.samplesContainer)
            samplesElements.corporaControl.addEventListener('change', (e) => {
                refreshList(drawSamplesList(nodeSamples, parseInt(e.target.value)))
            })
        }
        refreshList(samplesElements)

        clusterBlock.parentElement.classList.add('samples-opened')
        drawSamplesButton.innerHTML = 'Hide samples'
        clusterBlock.style.display = 'block'
        samplesOpen = true;
    }

    drawDiscriminativeSubstsButton.onclick = () => {
        if (discrSubstsOpen) {
            discriminativeSubstsContainer.innerHTML = ''
            drawDiscriminativeSubstsButton.innerHTML = 'Show discriminative substitutes'
            discrSubstsOpen = false;
            clusterBlock.parentElement.classList.remove('discr-substs-opened')
            return
        }
        discriminativeSubstsContainer.append(drawDiscriminativeSubstsSet(discriminativeSubsts))
        clusterBlock.parentElement.classList.add('discr-substs-opened')
        drawDiscriminativeSubstsButton.innerHTML = 'Hide discriminative substitutes'
        clusterBlock.style.display = 'block'
        discrSubstsOpen = true;
    }

    setSenseButton.onclick = () => {
        let sense = document.getElementById('senses-' + node.id).value
        let newSense = document.getElementById('new-sense-' + node.id).value
        setSense(node.id, newSense || sense)
    }

    clusterBlock.classList.add('cluster-block')

    refreshSenses(node, tags, [sensesList])

    if (isFinal) {
        addTag(tags, 'cluster', 'Cluster')
    }

    if (node.isImbalancedMerge) {
        addTag(tags, 'imbalance', 'Imbalanced merge')
    }

    if (node.highPMI) {
        addTag(tags, 'high-pmi', 'High PMI')
    }

    if (node.increasedEntropy) {
        addTag(tags, 'increased-entropy', 'Increased entropy')
    }

    return clusterBlock
}

const refreshSenses = (node, tags, sensesLists) => {
    sensesLists = sensesLists || document.querySelectorAll('.senses-list')
    tags = tags || document.getElementById('tags-' + node.id)

    if (tags) {
        removeTag(tags, 'sense')
        removeTag(tags, 'multi-sense')
        if (typeof dataset.senses[node.id] !== 'undefined') {
            if (!hasSense(node.id)) {
                addTag(tags, 'multi-sense', 'Sense: ' + dataset.senses[node.id])
            } else {
                addTag(tags, 'sense', 'Sense: ' + dataset.senses[node.id])
            }
        }
    }


    sensesLists.forEach(sensesList => {
        let nodeId = parseInt(sensesList.getAttribute('id').split('-')[1])

        sensesList.innerHTML = (!hasSense(nodeId) ? ['Select sense'] : []).concat(getSenses())
            .map(sense => '<option value="' + (sense === 'Select sense' ? '' : sense) + '" ' +
                (sense === dataset.senses[nodeId] ? 'selected' : '') + '>' + sense + '</option>').join('')
    })

}

const drawDiscriminativeSubstsSet = (substs) => {
    let container = document.createElement('div')
    let title = document.createElement('div')
    title.innerHTML = '<b>Discriminative substitutes</b><br />'
    container.append(title)
    substs.forEach((item) => {
        container.append(drawDiscriminativeSubstsItem(item))
    })
    return container
}

const addTag = (tags, cls, text) => {
    let tag = document.createElement('div')
    tag.innerHTML = text
    tag.classList.add('tag', cls)
    tags.append(tag)
}

const removeTag = (tags, cls) => {
    let tag = tags.querySelectorAll(".tag." + cls)
    if (tag.length > 0) {
        tag[0].remove()
    }
}

const getIteration = (clusterNodeId) => {
    return getClusterNodes().length - getClusterNodes().indexOf(clusterNodeId) + 1 - Object.keys(dataset.data).length
}

const getClusterSignature = (nodeIds) => {
    let substs = getSubstsPandPMI(nodeIds)
    substs.sort((a, b) => {
        return a[1] - b[1] || a[2] - b[2]
    }).reverse()

    return substs.slice(0, 10).map(s => s[0]).join(', ')
}

const getClusteringData = () => {
    return fetch(
        '/experiment/' + experiment_id + '/clustering/' + word + '/data',
        {
            method: 'GET'
        }
    )
        .then((response) => response.json())
        .then((data) => {
            dataset = data;
            warmUpClusteringTree(dataset.tree)
            return data
        });
}

const saveSenses = () => {
    return fetch(
        '/experiment/' + experiment_id + '/clustering/' + word + '/save-senses',
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(dataset.senses),
        }
    )
        .then((response) => response.json())
        .then((data) => {
            return data
        });
}

const plotCharts = () => {
    clusterCountsHistogram = new Chart(
        document.getElementById('counts-chart-container'),
        getCountsHistogramConfig()
    );
    vectorDistancesHistogram = new Chart(
        document.getElementById('vector-distances-chart-container'),
        getVectorDistancesConfig()
    );
    silARILineChart = new Chart(
        document.getElementById('sil-ari-chart-container'),
        getSilARIChartConfig()
    );
}
