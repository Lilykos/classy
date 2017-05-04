function getData() {
    return {
        labelsType: $('#label-type').val(),

        // text
        vectorizer: $('#vectorizer').val(),
        norm: $('#normalizer').val(),
        featureNumber: $('#numfeatures').val(),
        dimensionsNumber: $('#numdimensions').val(),
        stem: $('#stem').prop('checked'),
        stop: $('#stop').prop('checked'),

        // algorithms
        decomposition: $('#decomposition').val(),
        decompositionMetric: $('#decomposition-metric').val(),
        algorithms: $('#algorithms').val()
    }
}