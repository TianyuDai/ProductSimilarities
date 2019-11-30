# decrease the data file size
# filter out users have 0 item

# http://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data
import pickle
from parse import parse

ItemMapLocation = 'obj/steam_games_lite.pkl'

def getItemNames():
    result = {}
    datahandle = parse('../data/steam_games.json.gz')
    toremovekeys = [u"url", u"release_date", u"reviews_url"]
    with open('steam_games_lite.json', 'w') as fout:
        for line in datahandle:
            try:
                idkey = line[u"id"].encode("ascii")
            except KeyError:
                # lack id
                continue
            for removekey in toremovekeys:
                line.pop(removekey, None)
            fout.write(str(line))
            fout.write("\n")
            line.pop(u"id", None)
            result[idkey] = line
        # save object for further use
    with open(ItemMapLocation, 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    return result
    
def filterUserItem(allitems):
    datahandle = parse('../data/australian_users_items.json.gz')
    count = 0
    countgame = 0
    with open('australian_users_items_lite.json', 'w') as fout:
        for line in datahandle:
            thisuser = {"steam_id": line["steam_id"]}
            useritems = []
            for items in line["items"]:
                if items["item_id"] in allitems:
                    useritems.append({"id": items["item_id"], "playtime_forever": items["playtime_forever"]})
            if len(useritems) == 0:
                continue
            thisuser["items"] = useritems
            fout.write(str(thisuser))
            fout.write("\n")
            count += 1
            countgame += len(useritems)
    print("Total user: %d\nTotal items: %d" % (count, countgame))
 
    
def readItemMap():
    with open(ItemMapLocation, 'rb') as f:
        return pickle.load(f)
        
if __name__ == '__main__':
    
    allitems = getItemNames()
    #allitems = readItemMap()
    filterUserItem(allitems)