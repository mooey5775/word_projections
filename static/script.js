var ctx = document.getElementById('vector_chart').getContext('2d');

var scatterChart = new Chart(ctx, {
    type: 'scatter',
    data: {
        datasets: [{
            data: [{
                x: -10,
                y: 0
            }, {
                x: 0,
                y: 10
            }, {
                x: 10,
                y: 5
            }],
            labels: ["test", "test2", "test3"]
        }]
    },
    options: {
    	legend: {
    		display: false
    	},
        scales: {
            xAxes: [{
                type: 'linear',
                position: 'bottom',
                gridLines: {
                	color: '#333',
                	zeroLineColor: '#EEE'
                }
            }],
            yAxes: [{
            	gridLines: {
            		color: '#333',
            		zeroLineColor: '#EEE'
            	}
            }]
        },
        plugins: {
        	datalabels: {
        		color: 'white',
        		formatter: function(value, context) {
		          return context.dataset.labels[context.dataIndex];
		        },
		        anchor: 'start'
        	}
        }
    }
});

function getWord(form) {
	console.log(form.word.value);
	fetch("/get_vec/"+form.word.value)
	.then(res => res.json())
	.then(text => updateVector(text));
}

function updateVector(vec) {
	if (typeof vec === "number" && (vec % 1) === 0) {
		$("#vector").before('<div class="alert alert-danger alert-dismissible fade show" role="alert">'+
			'Invalid word'+
			'<button type="button" class="close" data-dismiss="alert" aria-label="Close">'+
				'<span aria-hidden="true">&times;</span>'+
			'</button>'+
		'</div>');
	} else {
		$("#vec_content").html(vec.join(", "));
		$("#vector").removeClass("d-none");
	}
}
