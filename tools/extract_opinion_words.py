import pprint
filename = '/vagrant/subjclueslen1-HLTEMNLP05.tff'
listwords = []
with open(filename, 'rb') as f:
    for line in f:
        i = line.find('word1=')
        words = line.split(' ')
        for w in words:
            if w.find('word1=') == 0:
                listwords.append(w[6:])
                break
pprint.pprint(listwords)
