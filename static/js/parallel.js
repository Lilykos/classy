function plot_parallel_coordinates(data) {
    function draw(d) {
        return line(dimensions.map(function(dimension) {
            return [x(dimension.name), dimension.scale(d[dimension.name])];
        }));
    }

    function moveToFront() { this.parentNode.appendChild(this); }

    function mouseover(d) {
        svg.classed("active", true);
        projection.classed("inactive", function(p) { return p !== d; });
        projection.filter(function(p) { return p === d; }).each(moveToFront);
        ordinal_labels.classed("inactive", function(p) { return p !== d.name; });
        ordinal_labels.filter(function(p) { return p === d.name; }).each(moveToFront);
    }

    function mouseout(d) {
        svg.classed("active", false);
        projection.classed("inactive", false);
        ordinal_labels.classed("inactive", false);
    }



    $('#parallel').empty();

    var margin = {top: 40, right: 40, bottom: 10, left: 120},
        width = $('#width-id').width() - 40 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;


    // DIMENSIONS ACCORDING TO LABELS - BINARY/MULTICLASS
    var dimensions_binary = [
        {
            name: "Name",
            scale: d3.scale.ordinal().rangePoints([0, height]),
            type: "string"
        }, {
            name: "Acc. (Train)",
            scale: d3.scale.linear().range([height, 0]),
            type: "number"
        },{
            name: "Acc. (Test)",
            scale: d3.scale.linear().range([height, 0]),
            type: "number"
        }, {
            name: "Prec",
            scale: d3.scale.linear().range([height, 0]),
            type: "number"
        }, {
            name: "Recall",
            scale: d3.scale.linear().range([height, 0]),
            type: "number"
        }, {
            name: "F1 Score",
            scale: d3.scale.linear().range([height, 0]),
            type: "number"
        }
    ];

        var dimensions_multiclass = [
        {
            name: "Name",
            scale: d3.scale.ordinal().rangePoints([0, height]),
            type: "string"
        }, {
            name: "Accuracy",
            scale: d3.scale.linear().range([height, 0]),
            type: "number"
        }, {
            name: "Prec (mi)",
            scale: d3.scale.linear().range([height, 0]),
            type: "number"
        }, {
            name: "Recall (mi)",
            scale: d3.scale.linear().range([height, 0]),
            type: "number"
        }, {
            name: "F1 Score (mi)",
            scale: d3.scale.linear().range([height, 0]),
            type: "number"
        }, {
            name: "Prec (ma)",
            scale: d3.scale.linear().range([height, 0]),
            type: "number"
        }, {
            name: "Recall (ma)",
            scale: d3.scale.linear().range([height, 0]),
            type: "number"
        }, {
            name: "F1 Score (ma)",
            scale: d3.scale.linear().range([height, 0]),
            type: "number"
        }
    ];

    // SELECT DIMENSIONS - BINARY/MULTICLASS
    var dimensions;
    if ($('#label-type').val() == 'binary') {
        dimensions = dimensions_binary;
    } else {
        dimensions = dimensions_multiclass;
    }


    var x = d3.scale.ordinal()
        .domain(dimensions.map(function(d) { return d.name; }))
        .rangePoints([0, width]);

    var line = d3.svg.line().defined(function(d) { return !isNaN(d[1]); });
    var yAxis = d3.svg.axis().orient("left");

    var svg = d3.select("#parallel")
        .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
        .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var dimension = svg.selectAll(".dimension")
        .data(dimensions)
        .enter()
        .append("g")
            .attr("class", "dimension")
            .attr("transform", function(d) { return "translate(" + x(d.name) + ")"; });

    dimensions.forEach(function(dimension) {
        dimension.scale.domain(dimension.type === "number"
            ? [0, 1]
            : data.map(function(d) { return d['Name']; }));
    });

    svg.append("g")
        .attr("class", "background")
        .selectAll("path")
        .data(data)
        .enter()
        .append("path")
            .attr("d", draw);

    svg.append("g")
        .attr("class", "foreground")
        .selectAll("path")
        .data(data)
        .enter()
        .append("path")
            .attr("d", draw);

    dimension.append("g")
        .attr("class", "axis")
        .each(function(d) { d3.select(this).call(yAxis.scale(d.scale)); })
        .append("text")
            .attr("class", "title")
            .attr("text-anchor", "middle")
            .attr("y", -15)
            .text(function(d) { return d.name; });

    var ordinal_labels = svg.selectAll(".axis text")
        .on("mouseover", mouseover)
        .on("mouseout", mouseout);

    var projection = svg.selectAll(".background path,.foreground path")
        .on("mouseover", mouseover)
        .on("mouseout", mouseout)
        .attr('class', 'tooltipped')
        .attr('data-tooltip', function (d) {
            return tmpl("tmpl-demo", d);
        });

    $('.tooltipped').tooltip({delay: 50, html: true});
}