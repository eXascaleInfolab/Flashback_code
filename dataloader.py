import os.path
import sys

from datetime import datetime

from dataset import PoiDataset, Usage

class PoiDataloader():
    ''' Creates datasets from our prepared Gowalla/Foursquare data files.
    The file consist of one check-in per line in the following format (tab separated):    
    
    <user-id> <timestamp> <latitude> <longitude> <location-id> 
    
    Check-ins for the same user have to be on continous lines.
    Ids for users and locations are recreated and continous from 0.
    '''
    
    def __init__(self, max_users=0, min_checkins=0):
        ''' max_users limits the amount of users to load.
        min_checkins discards users with less than this amount of checkins.               
        '''
        
        self.max_users = max_users
        self.min_checkins = min_checkins
        
        self.user2id = {}
        self.poi2id = {}
        
        self.users = []
        self.times = []
        self.coords = []
        self.locs = []
    
    def create_dataset(self, sequence_length, batch_size, split, usage=Usage.MAX_SEQ_LENGTH, custom_seq_count=1):
        return PoiDataset(self.users.copy(),\
                          self.times.copy(),\
                          self.coords.copy(),\
                          self.locs.copy(),\
                          sequence_length,\
                          batch_size,\
                          split,\
                          usage,\
                          len(self.poi2id),\
                          custom_seq_count)
        
    
    def user_count(self):
        return len(self.users)
    
    def locations(self):
        return len(self.poi2id)
    
    def read(self, file):
        if not os.path.isfile(file):
            print('[Error]: Dataset not available: {}. Please follow instructions under ./data/README.md'.format(file))
            sys.exit(1)

        # collect all users with min checkins:
        self.read_users(file)
        # collect checkins for all collected users:
        self.read_pois(file)
    
    def read_users(self, file):        
        f = open(file, 'r')            
        lines = f.readlines()
    
        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:
                    self.user2id[prev_user] = len(self.user2id)
                #else:
                #    print('discard user {}: to few checkins ({})'.format(prev_user, visit_cnt))
                prev_user = user
                visit_cnt = 1
                if self.max_users > 0 and len(self.user2id) >= self.max_users:
                    break # restrict to max users
    
    def read_pois(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        
        # store location ids
        user_time = []
        user_coord = []
        user_loc = []
        
        prev_user = int(lines[0].split('\t')[0])
        prev_user = self.user2id.get(prev_user)
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if self.user2id.get(user) is None:
                continue # user is not of interrest
            user = self.user2id.get(user)
                        
            time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1, 1)).total_seconds() # unix seconds
            lat = float(tokens[2])
            long = float(tokens[3])
            coord = (lat, long)            

            location = int(tokens[4]) # location nr
            if self.poi2id.get(location) is None: # get-or-set locations
                self.poi2id[location] = len(self.poi2id)
            location = self.poi2id.get(location)
    
            if user == prev_user:
                # insert in front!
                user_time.insert(0, time)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
            else:
                self.users.append(prev_user)
                self.times.append(user_time)
                self.coords.append(user_coord)
                self.locs.append(user_loc)
                
                # resart:
                prev_user = user 
                user_time = [time]
                user_coord = [coord]
                user_loc = [location] 
                
        # process also the latest user in the for loop
        self.users.append(prev_user)
        self.times.append(user_time)
        self.coords.append(user_coord)
        self.locs.append(user_loc)
