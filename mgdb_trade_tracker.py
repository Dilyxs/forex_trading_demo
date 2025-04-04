from pymongo import MongoClient, errors

mongo_conec_str = "mongodb+srv://ADSAYAN:ADSAYAN@clusterfirst.1zrfovm.mongodb.net/?retryWrites=true&w=majority&appName=ClusterFirst"


class DataDB_tracker:

    FOREX_COLL = "five_days_tracker"
   
    def __init__(self):
        self.client = MongoClient(mongo_conec_str)
        self.db = self.client.Forex

    def test_connection(self):
        print(self.db.list_collection_names())

    def add_one(self, collection, ob):
        try:
            _ = self.db[collection].insert_one(ob)
        except errors.InvalidOperation as error:
            print("add_one error:", error)

    def add_many(self, collection, list_ob):
        try:
            _ = self.db[collection].insert_many(list_ob)
        except errors.InvalidOperation as error:
            print("add_many error:", error)

    def query_distinct(self, collection, key):
        try:            
            return self.db[collection].distinct(key)
        except errors.InvalidOperation as error:
            print("query_distinct error:", error) 

    def query_single(self, collection, **kwargs):
        try:            
            r = self.db[collection].find_one(kwargs, {'_id': 0})
            return r
        except errors.InvalidOperation as error:
            print("query_single error:", error)

    def query_all(self, collection, **kwargs):
        try:
            data = []
            r = self.db[collection].find(kwargs, {'_id': 0})
            for item in r:
                data.append(item)
            return data
        except errors.InvalidOperation as error:
            print("query_all error:", error)

    def delete_many(self, collection, **kwargs):
        try:
            _ = self.db[collection].delete_many(kwargs)
        except errors.InvalidOperation as error:
            print("delete_many error:", error)

    def delete_one(self, collection, **kwargs):
        try:            
            _ = self.db[collection].delete_one(kwargs)
        except errors.InvalidOperation as error:
            print("delete_one error:", error)

    def delete_one_filter(self, collection, filter_query):
        try:
            self.db[collection].delete_one(filter_query)
            print(f"Successfully deleted document matching {filter_query}")
        except errors.InvalidOperation as error:
            print("delete_one error:", error)

    def update_one(self, collection, filter_query, update_key, update_value):
        try:
            self.db[collection].update_one(
                filter_query, 
                {"$set": {update_key: update_value}} 
            )
        except errors.InvalidOperation as error:
            print("update_one error:", error)
