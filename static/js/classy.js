$(function() {
    var textFile;
    var labelsFile;
    $('#text-input').on('change', function() { textFile = this.files[0]; });
    $('#labels-input').on('change', function() { labelsFile = this.files[0]; });


    // init materialize stuff
    $('select').material_select();
    $('#loading-screen').modal({dismissible: false, opacity: .6, endingTop: '40%'});
    $('#edit-options').modal({dismissible: true, opacity: .6});


    // init data load
    $('#load-data').on('click', function(e) {
        if (!textFile || !labelsFile) {
            Materialize.toast('Missing input file (Text and/or labels)!', 4000);
            return;
        }

        var formData = new FormData();
        formData.append('textFile', textFile);
        formData.append('labelsFile', labelsFile);

        // Load the file and render the input boxes.
        $.ajax({type: 'POST', url: '/load_data', data: formData,
            cache: false, contentType: false, processData: false,
            success: function() {
                $('.tab').removeClass('disabled grey lighten-3');
                $('.card-action').removeClass('hide');
            }
        });
    });


    // init run
    $('#run-classification').on('click', function() {
        $('#loading-screen').modal('open');

        var formData = new FormData();
        formData.append('data', JSON.stringify(getData()));

        $.ajax({type: 'POST', url: '/classify', data: formData,
            cache: false, contentType: false, processData: false,
            success: function(data) {
                console.log('Classification results.');
                console.log(data);

                // parallel coordinates (and rest of the plots)
                plot_parallel_coordinates(data.results);
                renderPlots(data);

                // show everything and remove loading screen modal
                unhideCards();
                $(function() { $('#loading-screen').modal('close');});
            }
        });
    });

});