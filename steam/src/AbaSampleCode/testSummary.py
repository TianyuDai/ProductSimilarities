# browse data

import os
from parse import parse

# http://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data

def testSummary():
    datahandle = parse('../../data/australian_users_items.json.gz')
    count = 0
    with open('summary.json', 'w') as fout:
        for line in datahandle:
            summaryfile = { your_key: line[your_key] for your_key in ['steam_id', 'items_count'] }
            playtime_forever = 0
            for items in line['items']:
                playtime_forever += items['playtime_forever']
            summaryfile['playtime_forever'] = playtime_forever
            fout.write(str(summaryfile))
            fout.write('\n')
            count += 1
            if count%2000 == 0:
                print(count)
    print('Total user: %d' % (count))
if __name__ == '__main__':
    testSummary()