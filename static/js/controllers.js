function sigmoid(x) {
	return Math.tanh(x)
}

function dsigmoid(y) {
	return 1.0 - Math.pow(y, 2.0);
}

function makeMatrix(I, J, fill) {
	m = [];
	for (var i = 0; i < I; i++) {
		n = [];
		for (var j = 0; j < J; j++) {
			n.push(fill);
		}
		m.push(n);
	}
	return m;
}

function NN(ni, nh, no) {
	// number of inputs, hidden, and output nodes
	this.ni = ni + 1; // +1 for bias node
	this.nh = nh;
	this.no = no;
	this.nl = nh.length + 2;
	
	// activations for nodes
	this.activation = [];
	row = [];
	for (var i = 0; i < this.ni; i++) {
		row.push(1.0);
	}
	this.activation.push(row)
	for (var i = 0; i < this.nl - 2; i++) {
		row = [];
		for (var j = 0; j < this.nh[i]; j++) {
			row.push(1.0);
		}
		this.activation.push(row);
	}
	row = [];
	for (var i = 0; i < this.no; i++) {
		row.push(1.0);
	}
	this.activation.push(row);

	this.weights = [];
	for (var i = 1; i < this.nl; i++) {
		this.weights.push(makeMatrix(this.activation[i-1].length, this.activation[i].length));
	}

	// this.ai = []
	// for (var i = 0; i < this.ni; i++) {
	// 	this.ai.push(1.0);	
	// }
	// this.ah = []
	// for (var i = 0; i < this.nh; i++) {
	// 	this.ah.push(1.0);	
	// }
	// this.ao = []
	// for (var i = 0; i < this.no; i++) {
	// 	this.ao.push(1.0);	
	// }
	
	// // create weights
	// this.wi = makeMatrix(this.ni, this.nh, 0.0);
	// this.wo = makeMatrix(this.nh, this.no, 0.0);
	// // set them to random values
	// for (var i = 0; i < this.ni; i++) {
	// 	for (j = 0; j < this.nh; j++) {
	// 		this.wi[i][j] = (Math.random() * 0.4) - 0.2;	
	// 	}
	// }
	// for (var j = 0; j < this.nh; j++) {
	// 	for (var k = 0; k < this.no; k++) {
	// 		this.wi[j][k] = (Math.random() * 4.0) - 2.0;
	// 	}
	// }
	
	// // last change in weights for momentum
	// this.ci = makeMatrix(this.ni, this.nh, 0.0);
	// this.co = makeMatrix(this.nh, this.no, 0.0);
	
	this.update = function(inputs) {
		if (inputs.length != this.ni - 1) {
			console.log("wrong number of inputs");
		}

		// input activation
		for (var i = 0; i < this.ni - 1; i++) {
			this.activation[0][i] = inputs[i];
		}

		for (var l = 1; l < this.nl; l++) {
			for (var j = 0; j < this.activation[l].length; j++) {
				var sum = 0.0;
				for (var i = 0; i < this.activation[l-1].length; i++) {
					sum = sum + this.activation[l-1][i] * this.weights[l-1][i][j];
				}
				this.activation[l][j] = sigmoid(sum)
			}
		}
		return this.activation[this.activation.length - 1]
	}
	
	this.backPropagate = function(targets, N, M) {
		if (targets.length != this.no) {
			console.log("wrong numbre of target values");	
		}
		
		// calculate error terms for output
		output_deltas = []
		for (var k = 0; k < this.no; k++) {
			output_deltas.push(0.0);	
		}
		for (var k = 0; k < this.no; k++) {
			error = targets[k] - this.ao[k];
			output_deltas[k] = 	dsigmoid(this.ao[k]) * error;
		}
		
		// calculate error terms for hidden
		hidden_deltas = []
		for (var j = 0; j < this.nh; j++) {
			hidden_deltas.push(0.0);	
		}
		for (var j = 0; j < this.nh; j++) {
			error = 0.0;
			for (var k = 0; k < this.no; k++) {
				error = error + output_deltas[k] * this.wo[j][k];	
			}
			hidden_deltas[j] = dsigmoid(this.ah[j]) * error;
		}
		
		// update output weights
		for (var j = 0; j < this.nh; j++) {
			for (var k = 0; k < this.no; k++) {
				change = output_deltas[j] * this.ah[j];
				this.wo[j][k] = this.wo[j][k] + N * change + M * this.co[i][j];
				this.co[j][k] = change;
			}
		}
		
		// update input weights
		for (var i = 0; i < this.ni; i++) {
			for (var j = 0; j < this.nh; j++) {
				change = hidden_deltas[j] * this.ai[i]	;
				this.wi[i][j] = this.wi[i][j] + N * change + M * this.ci[i][j];
				this.ci[i][j] = change
			}
		}
		
		// calculate error
		error = 0.0
		for (var k = 0; k < targets.length; k++) {
			error = error + 0.5 * Math.pow((targets[k] - this.ao[k]), 2.0);
		}
		return error;
	}
	
	this.tests = function(patterns) {
		for (var i = 0; i < patterns.length; i++) {
			var p = patterns[i];
			console.log(p[0] + " -> " + this.update(p[0]) + " -> " + p[1]);
		}
	}
	
	this.weights = function() {
		console.log("Input weights:");
		for (var i = 0; i < this.ni; i++) {
			console.log(this.wi[i]);	
		}
		console.log();
		console.log("Output weights:");
		for (var j = 0; j < this.nh; j++) {
			console.log(this.wo[j]);	
		}
	}
	
	this.train = function(patterns, iterations, N, M) {
		// N: learning rate
		// M: momentum factor
		for (var i = 0; i < iterations; i++) {
			var error = 0.0;
			for (var j = 0; j < patterns.length; j++) {
				var p = patterns[j];
				inputs = p[0];
				targets = p[1];
				this.update(inputs);
				error = error + this.backPropagate(targets, N, M);
			}
			if (i % 100 == 0) {
				console.log("error " + error);	
			}
		}
	}
}

function sorter(tests) {
	return tests;
	if (tests.length <= 1) {
		return tests;	
	}
	var left = [];
	var right = [];
	for (var i = 1; i < tests.length; i++) {
		if (tests[i]['outputs'][0] < tests[0]['outputs'][0]) {
			left.push(tests[i]);
		} else {
			right.push(tests[i]);
		}
	}
	return sorter(left).concat(tests[0]).concat(sorter(right));
}

mergeSort = function(items) {
	return mergeSortHelper(items);
}

mergeSortHelper = function(items) {
	// Terminal case: 0 or 1 item arrays don't need sorting
	if (items.length < 2) {
		return items;
	}

	var middle = Math.floor(items.length / 2),
		left    = items.slice(0, middle),
		right   = items.slice(middle);
	
	return this.merge(this.mergeSortHelper(left), this.mergeSortHelper(right));
}
merge = function(left, right) {
	var result  = [],
	il = 0,
	ir = 0;

	while (il < left.length && ir < right.length){
		if (left[il]['outputs'][0] < right[ir]['outputs'][0]){
			result.push(left[il++]);
		} else {
			result.push(right[ir++]);
		}
	}

	return result.concat(left.slice(il)).concat(right.slice(ir));
}

function convertNormalizedGREVerbaltoActual(normalizedScore, minimax) {
	return (minimax[1] - minimax[0]) * normalizedScore + minimax[0];	
}

function convertNormalizedGREWritingtoActual(normalizedScore, minimax) {
    return (minimax[3] - minimax[2]) * normalizedScore + minimax[2];
}

function convertNormalizedGREQuanttoActual(normalizedScore, minimax) {
    return (minimax[5] - minimax[4]) * normalizedScore + minimax[4];
}

function convertNormalizedGPAtoActual(normalizedScore, minimax) {
    return (minimax[7] - minimax[6]) * normalizedScore + minimax[6];
}


angular.module('app', [])

.controller('FormCtrl', ['$scope', '$log', '$http', function($scope, $log, $http) {
	Unknown = 0
	Other = 0
	International = 0
	US_Degree = 0.5
	American = 1
	
	$scope.inputs = {}
    $scope.outputs = {}
	
	$scope.ready = false;
	$scope.submitting = false;

	$scope.hasPrediction;
	
	$http.get("/inputs").success(function(data) { 
			$scope.inputs = data.response
	});
	$http.get("/outputs").success(function(data) { 
			$scope.outputs = data.response
	});


	function checkPredict() {
		$http.get('/predict/' + $scope.profile.school.id + "/" + $scope.profile.field.id).success(function(response) {
			if (response.response == 200) {
				$scope.hasPrediction = true;
				return;
			} else {
				$scope.hasPrediction = false;
				return;
			}
		});
	}
	
	function denormalizeInput(data, minimax) {
		result = [];
		// console.log($scope.inputs)
		// console.log(minimax)
		for (var i = 0; i < $scope.inputs.length; i++) {
			result[i] = (minimax[$scope.inputs[i].id]['max'] - minimax[$scope.inputs[i].id]['min']) * data[i] + minimax[$scope.inputs[i].id]['min']
		}
		// console.log(minimax)
		return result;
		/*if (data[0] == 0) {
			result.push("MS");	
		} else {
			result.push("PHD");	
		}
		if (data[1] == 1) {
			result.push("A");
		} else if (data[1] == 0.5) {
			result.push("U");
		} else {
			result.push("I");
		}
		return result;*/
	}
	
	$scope.parsed = false;
	
	$http.get("/available").success(function(data) { 
			$scope.schools = data.schools;
			$scope.fields = data.fields;

			$scope.profile = {
				'school':$scope.schools[Object.keys($scope.schools)[0]],
				'field':$scope.fields[Object.keys($scope.fields)[0]],
				'program': $scope.programs[0],
				'status': $scope.statuses[0],
				'gre_verbal': 170,
				'gre_quant': 170,
				'gre_writing': 6.0,
				'gpa': 4.0
			}
			
			$scope.ready = true;
			
			checkPredict();
	});
	
	$scope.programs = [
		{'display':'Masters', 'value': "MS", 'num': 0}, 
		{'display':'Ph.D.', 'value': "PHD", 'num': 1}, 
	];
	
	$scope.statuses = [
		{'display': 'American', 'value': 1},
		{'display': 'International, with US degree', 'value': 0.5},
		{'display': 'International', 'value': 0}
	];
	
	$scope.genders = [
		{'display': 'Male', 'value': 1},
		{'display': 'Female', 'value': 0},
	];
	
	$scope.ranks = [
		{'display':'Top 5','value':1.0},
		{'display':'Top 10','value':0.75},
		{'display':'Top 20','value':0.5},
		{'display':'Top 50','value':0.25},
		{'display':'Above 50','value':0.0}
	];

	$scope.recs_from = [
		{'display': '1.0 researcher known to admission committee', 'value': 1.0},
		{'display': '0.85 ---', 'value': 0.85},
		{'display': '0.70 professor in same country or community', 'value': 0.70},
		{'display': '0.55 ---', 'value': 0.55},
		{'display': '0.40 professor in a distant country', 'value': 0.40},
		{'display': '0.25 ---', 'value': 0.25},
		{'display': '0.10 nobody special', 'value': 0.10}
	];

	$scope.recs_observing = [
		{'display': '3.0 independent, self-directed researcher', 'value': 1.0},
		{'display': '2.6 ---', 'value': 0.867},
		{'display': '2.2 technical worker or apprentice researcher', 'value': 0.733},
		{'display': '1.8 ---', 'value': 0.600},
		{'display': '1.4 worker, organizer, volunteer', 'value': 0.467},
		{'display': '1.0 student in classroom', 'value': 0.333},
		{'display': '0.6 ---', 'value': 0.200}
	];

	$scope.recs_describing = [
		{'display': '4.0 future hero in field, unbelievably good', 'value': 1.0},
		{'display': '3.5 ---', 'value': 0.875},
		{'display': '3.0 future strong graduate student', 'value': 0.75},
		{'display': '2.5 ---', 'value': 0.625},
		{'display': '2.0 future good graduate student (this is typical)', 'value': 0.5},
		{'display': '1.5 ---', 'value': 0.375},
		{'display': '1.0 person with both good and bad points', 'value': 0.25},
		{'display': '0.5 ---', 'value': 0.125},
		{'display': '0.0 person not suited for graduate school', 'value': 0.0}
	];

	$scope.sop = [
		{'display': '3.0 exceptional, unusually strong', 'value': 1.0},
		{'display': '2.5 ---', 'value': 0.833},
		{'display': '2.0 good (typical)', 'value': 0.667},
		{'display': '1.5 ---', 'value': 0.5000},
		{'display': '1.0 uninformed, disorganized, or unrealistic', 'value': 0.333},
		{'display': '0.5 ---', 'value':  0.183},
		{'display': '0.0 multiply flawed', 'value': 0.0}
	];

	$scope.majors = [
		{'display': 'same as graduate major', 'value': 1.0},
		{'display': 'related to graduate major', 'value': 0.75},
		{'display': 'other', 'value': 0.25}
	]

	$scope.decisions = [
		{'display':'Admitted', 'value': 1},
		{'display':'Denied', 'value': 0}
	]
	
	$scope.americanOnly = false;
	$scope.predict = false;
	
	$scope.data;
	$scope.minimax;
	$scope.numApplied;
	$scope.numAccepted;
	$scope.numDenied;
	$scope.acceptanceRate;
	$scope.avg_application_gre_verbal;
	$scope.avg_application_gre_quant;
	$scope.avg_application_gre_writing;
	$scope.avg_application_gpa;
	$scope.avg_accepted_gre_verbal;
	$scope.avg_accepted_gre_quant;
	$scope.avg_accepted_gre_writing;
	$scope.avg_accepted_gpa;
	
	$scope.parse = function() {
		if ($scope.predict) {
			var	predict_option = "infered";
		} else {
			var predict_option = "standard";
		}
		console.log("/gradschool/" + $scope.profile.school.id + "/" + $scope.profile.field.id + "/" + predict_option)
		$http.get("/gradschool/" + $scope.profile.school.id + "/" + $scope.profile.field.id + "/" + predict_option).success(function(data) { 
			if ($scope.americanOnly) {
    			$scope.data = data[$scope.profile.program.value + "_American"];
			} else {
				$scope.data = data[$scope.profile.program.value + "_AllStatus"];
			}
			$scope.minimax = $scope.data['minimax'];
			$scope.sorted_tests = mergeSort($scope.data['tests']);
			$scope.numApplied = $scope.sorted_tests.length;
			$scope.numAccepted = 0.0;
			$scope.numDenied = 0.0;
			var totalGreVerbal = 0.0;
			var totalGreQuant = 0.0;
			var totalGreWriting = 0.0;
			var totalGpa = 0.0;
			var totalGreVerbalAccepted = 0.0;
			var totalGreQuantAccepted = 0.0;
			var totalGreWritingAccepted = 0.0;
			var totalGpaAccepted = 0.0;
			for (var i = 0; i < $scope.sorted_tests.length; i++) {
				var test = $scope.sorted_tests[i];
				var inputs = denormalizeInput(test['inputs'], $scope.minimax);
				if (test['expected'][0] == 1) {
					$scope.numAccepted++;
					totalGreVerbalAccepted += inputs[2];
					totalGreQuantAccepted += inputs[4];
					totalGreWritingAccepted += inputs[3];
					totalGpaAccepted += inputs[5];
				} 
				if (test['expected'][0] == 0) {
					$scope.numDenied++;	
				}
				totalGreVerbal += inputs[2];
				totalGreQuant += inputs[4];
				totalGreWriting += inputs[3];
				totalGpa += inputs[5];
				$scope.sorted_tests[i]['inputs'] = denormalizeInput(test['inputs'], $scope.minimax);
			}
			$scope.acceptanceRate = $scope.numAccepted / $scope.numApplied;
			$scope.avg_application_gre_verbal = totalGreVerbal / $scope.numApplied;
			$scope.avg_accepted_gre_verbal = totalGreVerbalAccepted / $scope.numAccepted;
			$scope.avg_application_gre_quant = totalGreQuant / $scope.numApplied;
			$scope.avg_accepted_gre_quant = totalGreQuantAccepted / $scope.numAccepted;
			$scope.avg_application_gre_writing = totalGreWriting / $scope.numApplied;
			$scope.avg_accepted_gre_writing = totalGreWritingAccepted / $scope.numAccepted;
			$scope.avg_application_gpa = totalGpa / $scope.numApplied;
			$scope.avg_accepted_gpa = totalGpaAccepted / $scope.numAccepted;
			$scope.test();
			$scope.parsed = true;
			checkPredict();
    	});   
	}
	
	$scope.score;
	$scope.admit;
	
	$scope.test = function() {
		var gre_verbal_n = ($scope.profile.gre_verbal - $scope.minimax['gre_verbal']['min']) / ($scope.minimax['gre_verbal']['max'] - $scope.minimax['gre_verbal']['min']);
		var gre_quant_n = ($scope.profile.gre_quant - $scope.minimax['gre_quant']['min']) / ($scope.minimax['gre_quant']['max'] - $scope.minimax['gre_quant']['min']);
		var gre_writing_n = ($scope.profile.gre_writing -$scope.minimax['gre_writing']['min']) / ($scope.minimax['gre_writing']['max'] - $scope.minimax['gre_writing']['min']);
		var gpa_n = ($scope.profile.gpa - $scope.minimax['gpa']['min']) / ($scope.minimax['gpa']['max'] - $scope.minimax['gpa']['min']);
		var status_n;
		if ($scope.americanOnly) {
			status_n = 1;	
		} else {
			status_n = $scope.profile.status.value;
		}
		var applicant = [$scope.profile.program.num,  status_n, gre_verbal_n, gre_writing_n, gre_quant_n, gpa_n];

		n = new NN(6, [8, 5, 3], 1);
		n.weights = $scope.data['weights'];
		console.log(n.weights)
		console.log(n.update(applicant))
		$scope.score = n.update(applicant)[0];

		accuracy = 0.2;
		index = 0;
		while (index < $scope.sorted_tests.length && $scope.sorted_tests[index].outputs[0] < $scope.score) {
			index++;
		}
		if ($scope.sorted_tests.length * accuracy > 1) {
			pointsToCollect = $scope.sorted_tests.length * accuracy;
		} else {
			pointsToCollect = $scope.sorted_tests.length;
		}
		dataPoints = [];
		offset = 0;
		while (dataPoints.length < pointsToCollect && offset < $scope.sorted_tests.length) {
			if ($scope.sorted_tests[index + offset]) {
				dataPoints.push($scope.sorted_tests[index + offset])
			}
			if ($scope.sorted_tests[index - offset] && offset != 0) {
				dataPoints.push($scope.sorted_tests[index - offset])
			}
			offset++;
		}
		admitted = 0;
		for (var i = 0; i < dataPoints.length; i++) {
			if (dataPoints[i].expected[0] == 1) {
				admitted++;
			}
		}
		$scope.admit = admitted / dataPoints.length;
	}

	$scope.range = function(num) {
    	return new Array(num);   
	}

	$scope.number_of_schools_submitting = 1;
	$scope.number_of_recs_submitting;
	$scope.submission = {
		'applications':[],
		'recs':[],
		'publications': false,
		'reu': false
	}
	
	$scope.removeSchool = function(index) {
		$scope.submission.applications.splice(index, 1);
		$scope.number_of_schools_submitting--;
	}

	$scope.removeRec = function(index) {
		$scope.submission.recs.splice(index, 1);
		$scope.number_of_recs_submitting--;
	}
	
	$scope.submit = function() {
		$http({
			url: '/submit',
			method: "POST",
			headers: {'Content-Type': 'application/json'},
			data: JSON.stringify($scope.submission)
		}).success(function(data) {
			console.log(data)
			$scope.number_of_schools_submitting = 1;
			$/*scope.submission = {
				'applications':[],
				'publications': false,
				'reu': false
			}
			*/
			$scope.parse()
		});
		$scope.submitting = false;
	}
	
}]);
