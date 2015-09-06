# Back-Propagation Neural Networks
#
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

import json
import os.path
import math

import urllib2
from bs4 import BeautifulSoup
from NeuralNetwork import NN

#settings.py
import os
# __file__ refers to the file settings.py
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')


gradSchools = {}
fields = {}
inputs = []
outputs = []

with open(os.path.join(APP_ROOT, 'data/schools.json'), 'rb')  as f:
    gradSchools = json.load(f)
with open(os.path.join(APP_ROOT, 'data/fields.json'), 'rb') as f:
    fields = json.load(f)
with open(os.path.join(APP_ROOT, 'data/inputs.json'), 'rb') as f:
    inputs = json.load(f)['response']
with open(os.path.join(APP_ROOT, 'data/outputs.json'), 'rb') as f:
    outputs = json.load(f)['response']

inputKeys = []
outputKeys = []
for inp in inputs:
    inputKeys.append(inp['id'])
for out in outputs:
    outputKeys.append(out['id'])

MS_only = 0
PHD_only = 1
MS_and_PHD = 2

AllStatus = 0
AmericanOnly = 1

Masters = 0
Phd = 1

Unknown = 0
Other = 0
International = 0
US_Degree = 0.5
American = 1

Rejected = 0
Accepted = 1
programs_2 = [
    {'id':'PHD', 'value': Phd},
    {'id':'MS', 'value': Masters}

]
programs = {
    'MS': MS_only,
    'PHD': PHD_only,
    'MS/PHD': MS_and_PHD
}

statuses = {
    'AllStatus': AllStatus,
    'American': AmericanOnly
}

smart_completion = [
    {'name':'standard', 'value': True},
    {'name':'infered', 'value': False}
]
class Parser:
    def getNeuralNet(this, school, field, predict, program, status):
        n = NN(len(inputKeys), [int(math.floor((len(inputKeys) + len(outputKeys)) * 5 / 4)), int(math.floor((len(inputKeys) + len(outputKeys)) * 3 / 4)), int(math.floor((len(inputKeys) + len(outputKeys)) * 2 / 4))], len(outputKeys))
        with open(os.path.join(APP_ROOT, 'data/schools/' + predict + "/" + school + '-' + field + '.json')) as infile:
            data = json.load(infile)
        weights = data[str(program) + '_' + str(status)]['weights']
        n.weights = weights
        return n

    def getPatterns(this, school, field, predict, program, status):
        with open(os.path.join(APP_ROOT, 'data/schools/' + predict + "/" + school + '-' + field + '.json')) as infile:
            data = json.load(infile)
        return data[str(program) + '_' + str(status)]['tests']

    # def savePatterns(this, school, field, predict, program, status, patterns):
    #     with open(os.path.join(APP_ROOT, 'data/schools/' + school + '-' + field + '_' + predict + '.json'), 'r') as f:
    #         data = json.load(f)
    #     with open('data/schools/' + school + '-' + field + '_' + predict + '.json', 'w') as f:
    #         data[str(program) + '_' + str(status)]['tests'] = patterns
    #         f.seek(0)
    #         json.dump(data, f, indent=4)

    def getMinimax(this, school, field, predict, program, status):
        with open(os.path.join(APP_ROOT, 'data/schools/' + predict + "/" + school + '-' + field + '.json')) as infile:
            data = json.load(infile)
        return data[str(program) + '_' + str(status)]['minimax']

    def updateMinimax(this, patterns):
        minimax = {}
        input_index = 0
        for inp in inputs:
            listOfInputValues = []
            for pattern in patterns:
                listOfInputValues.append(pattern[0][input_index])
            minimax[inp['id']] = {}
            if len(listOfInputValues) == 0:
                minimax[inp['id']]['max'] = 0
                minimax[inp['id']]['min'] = 0
            else:
                minimax[inp['id']]['max'] = max(listOfInputValues)
                minimax[inp['id']]['min'] = min(listOfInputValues)
            input_index += 1
        return minimax

    def denormalizeInput(this, pattern, minimax):
        result = []
        index = 0
        for inp in inputs:
            # print(inp)
            result.append((minimax[inp['id']]['max'] - minimax[inp['id']]['min']) * pattern[index] + minimax[inp['id']]['min'])
            index += 1
        return result

    def normalizeInputs(this, pattern, minimax):
        result = []
        index = 0
        for inp in inputs:
            if inp['id'] == 'ms_phd' or inp['id'] == 'status':
                result.append(float(pattern[index]))
            # print((minimax[inp['id']]['max'] - minimax[inp['id']]['min']) * pattern[index] + minimax[inp['id']]['min'])
            elif minimax[inp['id']]['max'] - minimax[inp['id']]['min'] != 0:
                result.append(float((pattern[index] - minimax[inp['id']]['min']))/float((minimax[inp['id']]['max'] - minimax[inp['id']]['min'])))
            else:
                result.append(1.0)
            index += 1
        return result

    def saveWeights(this, school, field, predict, program, status, weights):
        with open(os.path.join(APP_ROOT, 'data/schools/' + predict + "/" + school + '-' + field + '.json'), 'r') as f:
            data = json.load(f)
        with open(os.path.join(APP_ROOT, 'data/schools/' + predict +"/" + school + '-' + field + '.json'), 'w') as f:
            print(os.path.join(APP_ROOT, 'data/schools/' + predict +"/" + school + '-' + field + '.json'))
            data[str(program) + '_' + str(status)] = weights
            f.seek(0)
            json.dump(data, f, indent=4)

    def updatePatternsAndNeuralNet(this, school, field, predict, program, status, submission, application):
        patterns = this.getPatterns(school, field, predict, program, status)
        minimax = this.getMinimax(school, field, predict, program, status)
        converted_pattern = []
        for pattern in patterns:
            converted_pattern.append([this.denormalizeInput(pattern['inputs'], minimax), pattern['expected']])
        patterns = converted_pattern
        # print(patterns)
        # print(len(patterns))
        patterns.append([submission, [application['decision']]])
        # print(patterns)
        # update minimax
        minimax = this.updateMinimax(patterns)
        # normalize patterns
        patterns_n = []
        # print(patterns)
        for pattern in patterns:
            patterns_n.append([this.normalizeInputs(pattern[0], minimax), pattern[1]])
        patterns = patterns_n
        # update neural net
        n = this.getNeuralNet(school, field, predict, program, status)
        n.train(patterns, 1000, 0.4, 0.1)
        n.test(patterns)
        dump_data = {}
        tests = []
        for pat in patterns:
            test = {}
            test['inputs'] = pat[0]
            test['outputs'] = n.update(pat[0])
            test['expected']= pat[1]
            tests.append(test)
        weights = n.getWeights()
        dump_data['tests'] = tests
        dump_data['weights'] = weights
        dump_data['minimax'] = minimax
        this.saveWeights(school, field, predict, program, status, dump_data)

    def addSubmission(this, submission):
        for application in submission['applications']:
            school = application['school']
            field = application['field']
            program = application['ms_phd']
            submissionInputs = []
            for inp in inputKeys:
                if inp in submission:
                    submissionInputs.append(submission[inp])
                if inp in application:
                    submissionInputs.append(application[inp])
            # update and store patterns
            for option in smart_completion:
                if os.path.exists('data/schools/' + option['name'] + "/" + school + '-' + field + '.json'):
                    for p in programs_2:
                        if program == p['value']:
                            if submission['status'] == American:
                                this.updatePatternsAndNeuralNet(school, field, option['name'], p['id'], 'American', submissionInputs, application)
                            this.updatePatternsAndNeuralNet(school, field, option['name'], p['id'], 'AllStatus', submissionInputs, application)




import sys, getopt

def main():
   # Store input and output file names
    gradSchool = {}
    ofile=''
    add_all = False

    addAll()
#     import argparse
#
#     from optparse import OptionParser
#     parser = OptionParser()
#     parser.add_option("-s", "--add-school", dest="school",
#                       help="adds school to data", metavar="school")
#     parser.add_option("-f", "--add-field", dest="field",
#                       help="adds field to data", metavar="field")
#     parser.add_option("-q", "--query", dest="query",
#                       help="query to search", metavar="query")
#     parser.add_option("-a", "--add-all", action="store_true", dest="add_all", default=False,
#                       help="add all schools and fields")
#     # parser.add_option("-q", "--quiet",
#     #                   action="store_false", dest="verbose", default=True,
#     #                   help="don't print status messages to stdout")
#     # parser = argparse.ArgumentParser(description='Data Parsing Options.')
#     # parser.add_argument('-s', '--add-school', type=add_school, nargs=2, help="Adds school")
#     # parser.add_argument('-f', '--add-field', type=add_field, nargs=2, help="Adds field")
#     # parser.add_argument('-a', '--add-all', type=add_all, nargs=1, help="Adds all schools and fields")
#     # parser.add_argument('integers', metavar='N', type=int, nargs='+',
#     #                    help='an integer for the accumulator')
#     # parser.add_argument('--sum', dest='accumulate', action='store_const',
#     #                    const=sum, default=max,
#     #                    help='sum the integers (default: find the max)')
#
#     options, args = parser.parse_args()
#     options = vars(options)
#     if options['add_all']:
#         addAll()
#     elif options['school'] != None:
#         if options['query'] != None:
#             add_school(options['school'], options['query'])
#         else:
#             print("Query needed")
#     elif options['field'] != None:
#         if options['query'] != None:
#             add_field(options['field'], options['query'])
#         else:
#             print("Query needed")
#     else:
#         print("Invalid options")
#
#     # getAll()

def add_school(school, query):
    print("Adding " + school + " with query " + query)

def add_field(field, query):
    print("Adding " + field + " with query " + query)

def addAll():
    weightsSet = {}
    if not os.path.exists(os.path.join(APP_ROOT, 'data/available.json')):
        with open(os.path.join(APP_ROOT, 'data/available.json'),'w+') as f:
            gradSchoolsFiles = {'schools':{},'fields':{}}
            gradSchoolsFiles['fields'] = fields
            for field in fields:
                fields[field]['id'] = field
            f.seek(0)
            json.dump(gradSchoolsFiles, f, indent=4)

    for smart_completion_option in smart_completion:
        for school in gradSchools:
            for field in fields:
                if not os.path.exists(os.path.join(APP_ROOT, 'data/schools/' + smart_completion_option['name'] + '/' + school + '-' + field + '.json')):
                    school_data = Data(school, field, smart_completion_option['value'], inputKeys, outputKeys)
                    for program in programs:
                        for status in statuses:
                            pat = school_data.getPattern(programs[program], statuses[status])
                            n = NN(len(inputKeys), [int(math.floor((len(inputKeys) + len(outputKeys)) * 5 / 4)), int(math.floor((len(inputKeys) + len(outputKeys)) * 3 / 4)), int(math.floor((len(inputKeys) + len(outputKeys)) * 2 / 4))], len(outputKeys))
                            n.train(pat, 1000, 0.4, 0.1)
                            n.test(pat)

                            dump_data = {}
                            dump_data['weights'] = n.getWeights()
                            tests = []
                            for p in pat:
                                test = {}
                                test['inputs'] = p[0]
                                test['outputs'] = n.update(p[0])
                                test['expected']= p[1]
                                tests.append(test)
                            dump_data['tests'] = tests
                            dump_data['minimax'] = school_data.minimax
                            weightsSet[str(program) + '_' + str(status)] = dump_data

                        # dumps page information to json file
                        with open(os.path.join(APP_ROOT, 'data/schools/' + smart_completion_option['name'] + '/' + school + '-' + field + '.json'), 'w') as outfile:
                            json.dump(weightsSet, outfile, indent=4)
                        with open(os.path.join(APP_ROOT, 'data/schools/' + smart_completion_option['name'] + '/' + school + '-' + field + '.json')) as infile:
                            weightsSet = json.load(infile)

                    with open(os.path.join(APP_ROOT, 'data/available.json'), 'r') as f:
                        gradSchoolsFiles = json.load(f)
                    with open(os.path.join(APP_ROOT, 'data/available.json'), 'w') as f:
                        gradSchoolsFiles['schools'][school] = gradSchools[school]
                        gradSchoolsFiles['schools'][school]['id'] = school
                        if smart_completion_option['value']:
                            if 'fields' not in gradSchoolsFiles['schools'][school]:
                                 gradSchoolsFiles['schools'][school]['fields'] = []
                            gradSchoolsFiles['schools'][school]['fields'].append(field)
                        f.seek(0)
                        json.dump(gradSchoolsFiles, f, indent=4)
                else:
                    print('data/schools/' + smart_completion_option['name'] + '/' + school + '-' + field + '.json')

gre_verbal_conversion = {800:170, 790:170, 780:170, 770:170, 760:170, 750:169, 740:169, 730:168, 720:168, 710:167, 700:166, 690:165, 680:165, 670:164, 660:164, 650:163, 640:162, 630:162, 620:161, 610:160, 600:160, 590:159, 580:158, 570:158, 560:157, 550:156, 540:156, 530:155, 520:154, 510:154, 500: 153, 490:152, 480:152, 470:151, 460:151, 450:150, 440:149, 430:149, 420:148, 410:147, 400:146, 390:146, 380:145, 370:144, 360:143, 350:143, 340:142, 330:141, 320:140, 310:139, 300:138, 290:137, 280:135, 270:134, 260:133, 250:132, 240:131, 230:130, 220:130, 210:130, 200: 130}
gre_quant_conversion = {800:166, 790:164, 780:163, 770:161, 760:160, 750:159, 740:158, 730:157, 720:156, 710:155, 700:155, 690:154, 680:153, 670:152, 660:152, 650:151, 640:151, 630:150, 620:149, 610:149, 600:148, 590:148, 580:147, 570:147, 560:146, 550:146, 540:145, 530:145, 520:144, 510:144, 500: 144, 490:143, 480:143, 470:142, 460:142, 450:141, 440:141, 430:141, 420:140, 410:140, 400:140, 390:139, 380:139, 370:138, 360:138, 350:138, 340:137, 330:137, 320:136, 310:136, 300:136, 290:135, 280:135, 270:134, 260:134, 250:133, 240:133, 230:132, 220:132, 210:131, 200: 131}

class Data:
    def __init__(this, school, field, completeOnly, inputs, outputs):
        this.school = school
        this.field = field
        this.inputs = inputs
        this.outputs = outputs
        this.results = this.getData(completeOnly)
        this.minimax = this.getMinimax()
        this.processedResults = this.preprocess()
        this.pat = this.getPattern()
    def getData(this, completeOnly):
        query = gradSchools[this.school]['query'] + ' ' +  fields[this.field]['query']
        result = []
        numAdded = 0
        done = False
        page = 1
        totalPages = 999
        while not done and page <= totalPages:
            content = urllib2.urlopen("http://thegradcafe.com/survey/index.php?q=" + query + "&t=a&o=&p=" + str(page)).read()
            soup = BeautifulSoup(content);
            for div in soup.find_all('div'):
                divClass = div.get('class','')
                if "pagination" in divClass:
                    if ('over' in div.get_text()):
                        overIndex = div.get_text().index('over')
                        totalPages = div.get_text()[overIndex + 5:]
                        totalPages = totalPages.split()
                        totalPages = int(totalPages[0])
                    else:
                        totalPages = 0

            if "Bad Request" in soup.get_text():
                print("Bad Request 400")
                done = True

            for table in soup.find_all('table'):
                table_class = table.get('class','')
                if "results" in table_class:
                    entry = 0
                    for tr in table.find_all('tr'):
                        if entry > 0:
                            col = 0
                            data = {}
                            for td in tr.find_all('td'):
                                if col == 0 and "Institution" not in td.get_text():
                                    data['institution'] = td.get_text()
                                if col == 1 and "Program" not in td.get_text():
                                    data['program'] = td.get_text()
                                    if "PhD" in data['program']:
                                        data['ms_phd'] = Phd
                                    elif "Masters" in data['program']:
                                        data['ms_phd'] = Masters
                                    else:
                                        data['ms_phd'] = 0.5
                                if col == 2 and "Decision" not in td.getText():
                                    if "Accepted" in td.get_text():
                                        data['decision'] = Accepted
                                    else:
                                        data['decision'] = Rejected
                                    spanNum = 0
                                    for span in td.find_all('span'):
                                        if spanNum == 1:
                                            try:
                                                data['gpa'] = float(span.get_text()[15:19])
                                            except ValueError:
                                                pass
                                            try:
                                                if ((float(span.get_text()[40:43]) >= 130 and float(span.get_text()[40:43]) <= 170)) or (float(span.get_text()[40:43]) >= 200 and float(span.get_text()[40:43]) <= 800):
                                                    data['gre_verbal'] = float(span.get_text()[40:43])
                                                    if (data['gre_verbal'] > 170):
                                                        while data['gre_verbal'] % 10 != 0:
                                                            data['gre_verbal'] -= 1
                                                        print("Old Verbal GRE score: " + str(data['gre_verbal']) + " -> " + str(gre_verbal_conversion[data['gre_verbal']]))
                                                        data['gre_verbal'] = gre_verbal_conversion[data['gre_verbal']]
                                            except ValueError:
                                                pass
                                            try:
                                                if ((float(span.get_text()[44:47]) >= 130 and float(span.get_text()[44:47]) <= 170)) or (float(span.get_text()[44:47]) >= 200 and float(span.get_text()[44:47]) <= 800):
                                                    data['gre_quant'] = float(span.get_text()[44:47])
                                                    if (data['gre_quant'] > 170):
                                                        while data['gre_quant'] % 10 != 0:
                                                            data['gre_quant'] -= 1
                                                        print("Old Quant GRE score: " + str(data['gre_quant']) + " -> " + str(gre_quant_conversion[data['gre_quant']]))
                                                        data['gre_quant'] = gre_quant_conversion[data['gre_quant']]
                                            except ValueError:
                                                pass
                                            try:
                                                data['gre_writing'] = float(span.get_text()[48:52])
                                            except ValueError:
                                                pass
                                        spanNum += 1
                                if col == 3 and "St" not in td.get_text():
                                    if "O" in td.get_text():
                                        data['status'] = Other
                                    elif "I" in td.get_text():
                                        data['status'] = International
                                    elif "U" in td.get_text():
                                        data['status'] = US_Degree
                                    elif "A" in td.get_text():
                                        data['status'] = American
                                    else:
                                        data['status'] = Unknown
                                if col == 4 and "Date Added" not in td.get_text():
                                    if "2009" in td.get_text():
                                        done = True
                                col += 1
                            if (completeOnly and len(data) == 9) or not completeOnly:
                                result.append(data)
                                numAdded += 1
                                print(str(numAdded) + " entries added for " + this.school + '-' + this.field + ".")
                        entry += 1
                page += 1

        return result
    def getMinimax(this):
        minimax = {}
        for inp in this.inputs + this.outputs:
            minimax[inp] = {}
            listOfInputValues = []
            for entry in this.results:
                if inp in entry:
                    listOfInputValues.append(entry[inp])
            if len(listOfInputValues) == 0:
                minimax[inp]['max'] = 0
                minimax[inp]['min'] = 0
            else:
                minimax[inp]['max'] = max(listOfInputValues)
                minimax[inp]['min'] = min(listOfInputValues)
        return minimax

    # normalizes data and fills in missing data using a neural net
    def preprocess(this):
        inputKeys = this.inputs + this.outputs
        # Normalize values
        for entry in this.results:
            for inp in inputKeys:
                if inp in entry:
                    if this.minimax[inp]['max'] - this.minimax[inp]['min'] != 0:
                        entry[inp] = (entry[inp] - this.minimax[inp]['min']) / (this.minimax[inp]['max'] - this.minimax[inp]['min'])
                    else:
                        entry[inp] = 1
                    if entry[inp] > 1 or entry[inp] < 0:
                        print("ERROR")

        # Fills in missing data
        completeEntrys = []
        for entry in this.results:
            containsAllKeys = True
            for key in inputKeys:
                if key not in entry:
                    containsAllKeys = False
            if containsAllKeys:
                completeEntrys.append(entry)
        processedData = []
        numProcessed = 0
        for entry in this.results:
            missingKeys = []
            presentKeys = []
            for key in inputKeys:
                if key not in entry:
                    missingKeys.append(key)
                else:
                    presentKeys.append(key)
            if len(missingKeys) > 0:
                i = []
                for presentKey in presentKeys:
                    i.append(entry[presentKey])
                trainingData = []
                for completeEntry in completeEntrys:
                    trainingRow = []
                    inputs = []
                    outputs = []
                    for presentKey in presentKeys:
                        inputs.append(completeEntry[presentKey])
                    for missingKey in missingKeys:
                        outputs.append(completeEntry[missingKey])
                    trainingRow.append(inputs)
                    trainingRow.append(outputs)
                    trainingData.append(trainingRow)

                n = NN(len(presentKeys), [int(math.floor((len(presentKeys) + len(missingKeys)) * 3 / 4)), int(math.floor((len(presentKeys) + len(missingKeys)) * 2 / 4)), int(math.floor((len(presentKeys) + len(missingKeys)) * 1 / 4))], len(missingKeys))
                n.train(trainingData)
                n.test(trainingData)
                predictions = n.update(i)
                index = 0
                for missingKey in missingKeys:
                    entry[missingKey] = predictions[index]
                    index += 1
            processedData.append(entry)
            numProcessed += 1
            print(str(numProcessed) + " entries processed for " + this.school + '-' + this.field + ".")
        return processedData


    def getPattern(this, filt=MS_only, status=AllStatus):
        # inputs [ms_phd, status, gpa, gre_verbal, gre_quant, gre_writing]
        # output [decision]
        inputKeys = this.inputs
        result = []
        for entry in this.processedResults:
            if filt == MS_only and status == AmericanOnly:
                if entry['ms_phd'] == Masters and entry['status'] == American:
                    resultRow = []
                    inputs = []
                    outputs = []
                    for inputKey in inputKeys:
                        inputs.append(entry[inputKey])
                    outputs.append(entry['decision'])
                    resultRow.append(inputs)
                    resultRow.append(outputs)
                    result.append(resultRow)
            elif filt == MS_only:
                if entry['ms_phd'] == Masters:
                    resultRow = []
                    inputs = []
                    outputs = []
                    for inputKey in inputKeys:
                        inputs.append(entry[inputKey])
                    outputs.append(entry['decision'])
                    resultRow.append(inputs)
                    resultRow.append(outputs)
                    result.append(resultRow)
            elif filt == PHD_only and status == AmericanOnly:
                if entry['ms_phd'] == Phd and entry['status'] == American:
                    resultRow = []
                    inputs = []
                    outputs = []
                    for inputKey in inputKeys:
                        inputs.append(entry[inputKey])
                    outputs.append(entry['decision'])
                    resultRow.append(inputs)
                    resultRow.append(outputs)
                    result.append(resultRow)
            elif filt == PHD_only:
                 if entry['ms_phd'] == Phd:
                    resultRow = []
                    inputs = []
                    outputs = []
                    for inputKey in inputKeys:
                        inputs.append(entry[inputKey])
                    outputs.append(entry['decision'])
                    resultRow.append(inputs)
                    resultRow.append(outputs)
                    result.append(resultRow)
            elif filt == MS_and_PHD and status == AmericanOnly:
                if entry['ms_phd'] == Phd and entry['status'] == American:
                    resultRow = []
                    inputs = []
                    outputs = []
                    for inputKey in inputKeys:
                        inputs.append(entry[inputKey])
                    outputs.append(entry['decision'])
                    resultRow.append(inputs)
                    resultRow.append(outputs)
                    result.append(resultRow)
            else:
                resultRow = []
                inputs = []
                outputs = []
                for inputKey in inputKeys:
                    inputs.append(entry[inputKey])
                outputs.append(entry['decision'])
                resultRow.append(inputs)
                resultRow.append(outputs)
                result.append(resultRow)
        return result


if __name__ == '__main__':
    main()
