let clusterCountsHistogram = null;
let vectorDistancesHistogram = null;
let silARILineChart = null;

const getExamplesData = () => {
    return fetch(
        '/experiment/' + experiment_id + '/examples/' + word + '/data',
        {
            method: 'GET'
        }
    )
        .then((response) => response.json())
        .then((data) => {
            dataset = data
            return data
        });
}

const showClusterInfo = (cluster) => {
    let container = document.getElementById('samples-container')
    let nodeIds = Object.keys(dataset.data).filter((nodeId) => dataset.data[nodeId].cluster === cluster || cluster === -1)
    let topSubsts = drawMultiNodeSelectionSummaryTable(nodeIds)
    let examples = document.createElement('div')
    let combined = document.createElement('div')
    let summary = document.createElement('div')

    let nodes = nodeIds.map((nodeId) => {
        return {
            id: nodeId,
            data: dataset.data[nodeId]
        }
    })

    examples.style.overflow = 'scroll';
    examples.style.padding = '10px';
    examples.height = '1000px'

    let totals = document.createElement('span')
    totals.innerHTML = 'Total samples: <b>' + nodeIds.length + '</b>&nbsp;'

    let corpus1Samples = nodes.filter((node) => node.data.corpus === 0)
    totals.innerHTML += '<a href="#corpus1Start">' + dataset.corpora_names[0] + '</a>: <b>' + corpus1Samples.length + '</b>&nbsp;'

    let corpus2Samples = nodes.filter((node) => node.data.corpus === 1)
    if (corpus2Samples.length > 0)
        totals.innerHTML += '<a href="#corpus2Start">' + dataset.corpora_names[1] + '</a>: <b>' + corpus2Samples.length + '</b>'

    let clusterPmi = (cluster >= 0 ? dataset.cluster_pmis[cluster] : 0)
    let corporaName = (cluster >= 0 ? dataset.corpora_names[0] : 'Any corpora')
    let clusterInfo = 'Cluster ' + corporaName

    if (cluster > 0) {
         clusterInfo += ', PMI(' + dataset.corpora_names[0] + ', Cluster) = ' + clusterPmi[0]
         clusterInfo += ', PMI(' + dataset.corpora_names[1] + ', Cluster) = ' + clusterPmi[1]
    }

    summary.append(clusterInfo)
    summary.append(document.createElement('br'))
    summary.append(totals)
    summary.append(document.createElement('br'))
    summary.append('Number of substitutes to show ')
    let sampleElements1 = drawSamplesList(corpus1Samples)
    summary.append(sampleElements1.nSubstsControl)

    examples.append(summary)
    examples.append(document.createElement('br'))
    let corpus1Start = document.createElement('div')
    corpus1Start.id = 'corpus1Start'
    corpus1Start.style.fontSize = '18px;'
    corpus1Start.innerHTML = dataset.corpora_names[0]
    examples.append(corpus1Start)

    examples.append(sampleElements1.samplesContainer)

    let corpus2Start = document.createElement('div')
    corpus2Start.id = 'corpus2Start'
    corpus2Start.style.fontSize = '18px;'
    corpus2Start.innerHTML = dataset.corpora_names[1]
    if (corpus2Samples.length > 0)
        examples.append(corpus2Start)

    let sampleElements2 = drawSamplesList(corpus2Samples)
    examples.append(sampleElements2.samplesContainer)

    combined.append(topSubsts)
    combined.append(examples)
    container.replaceChildren(combined)
    drawSubsts(nodeIds, 0)
}

document.addEventListener('DOMContentLoaded', () => {
    toggleProcessingOverlay()
    getExamplesData().then(() => {
        setTimeout(() => {
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
            toggleProcessingOverlay()
        }, 0)
        showClusterInfo(-1)
    });

    document.getElementById('counts-chart-container').onclick = (evt) => {
        let bar = clusterCountsHistogram.getElementsAtEventForMode(evt, 'nearest', {intersect: true}, false)
        if (bar.length === 1) {
            let cluster = bar[0].index
            let corpus = bar[0].datasetIndex
            showClusterInfo(cluster)
        }
    }
});



