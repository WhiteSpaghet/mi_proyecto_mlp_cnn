import firebase_admin
from firebase_admin import credentials, storage, firestore

def init_firebase(service_account_path, bucket_name=None):
    cred = credentials.Certificate(service_account_path)
    options = {}
    if bucket_name:
        options['storageBucket'] = bucket_name
    app = firebase_admin.initialize_app(cred, options)
    db = firestore.client(app)
    bucket = storage.bucket(app=app) if bucket_name else None
    return db, bucket

def upload_model_to_storage(bucket, model_path, dest_name):
    blob = bucket.blob(dest_name)
    blob.upload_from_filename(model_path)
    blob.make_public()
    return blob.public_url

def save_metrics_to_firestore(db, collection_name, data):
    db.collection(collection_name).add(data)