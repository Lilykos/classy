function getData() {
    return {
        labelsType: $('#label-type').val(),

        // text
        embedding: $('#embedding').val(),
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

function renderPlots(data) {
    $('#confusion-matrix').html(data.confusion_matrix);
    $('#prec-rec').html(data.precrec);
    $('#roc-curves').html(data.roc);
}

function unhideCards() {
    var selectors = [
        '#parallel-container',
        '#confusion-matrix-container',
        '#precision-recall-container',
        '#roc-curves-container'
    ];
    selectors.forEach(function (sel) { $(sel).removeClass('hide'); });
    $('.carousel').carousel();
}