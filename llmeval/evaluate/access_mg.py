import requests
import json

import requests

# def download_mongodb_dump(url, output_file, headers, data):
#     """
#     Downloads a MongoDB dump from the given URL and saves it to the specified file.

#     Parameters:
#     url (str): The URL to download the MongoDB dump from.
#     output_file (str): The local file path to save the downloaded dump.
#     """
#     response = requests.get(url, stream=True, headers=headers, data=data)
#     response.raise_for_status()  # Check if the request was successful

#     with open(output_file, 'wb') as file:
#         for chunk in response.iter_content(chunk_size=8192):
#             file.write(chunk)
    
#     print(f"MongoDB dump downloaded successfully and saved to {output_file}")

# if __name__ == "__main__":
#     # Replace this URL with your actual endpoint that serves the MongoDB dump
#     #MONGODB_DUMP_URL = 'http://example.com/download/mongodb-dump'
#     OUTPUT_FILE = 'mongodb-dump.tar.gz'  # Adjust the file name and extension as needed
#     MONGODB_DUMP_URL = 'mongodb+srv://q1:8azjYw297Szi1ddz@q.akhuf.mongodb.net/?retryWrites=true&w=majority'
#     url = 'https://us-west-2.aws.data.mongodb-api.com/app/data-sqvegnw/endpoint/data/v1'# "https://us-west-2.aws.data.mongodb-api.com/app/data-sqvegnw/endpoint/data/v1/"
#     payload = json.dumps({
#         "collection": "lvm-log",
#         "database": "lvm-logs-server",
#         "dataSource": "q",
#         "projection": {
#             "_id": 1
#         }
#     })
#     headers = {
#     'Content-Type': 'application/json',
#     'Access-Control-Request-Headers': '*',
#     'api-key': 'AIpCJQLVPRFqlOeTVo3xPQ2RWI4LJpn1lClwigUKRPLUrsYjkWHQhOA0Ljli3INA',
#     }

#     response = requests.request("GET", url, headers=headers, data=payload)
#     #download_mongodb_dump(url=url, output_file=OUTPUT_FILE, headers=headers, data=payload)

#     print(response.text)


from pymongo import MongoClient

def fetch_data():
    # Replace the following with your MongoDB connection string
    uri = 'mongodb+srv://q1:8azjYw297Szi1ddz@q.akhuf.mongodb.net/?retryWrites=true&w=majority'

    # Create a MongoClient instance
    client = MongoClient(uri)

    # Specify the database and collection
    database = client['lvm-logs-server']  # Replace with your database name
    collection = database['lvm-logs']  # Replace with your collection name

    #try:
        # Find all documents in the collection
    documents = collection.find_one()

        # Iterate over the documents and print them
    for doc in documents:
            print(doc)

    #finally:
        # Close the connection
        #client.close()

if __name__ == '__main__':
    fetch_data()


