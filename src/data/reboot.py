from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["MAR25_CMLOPS_RAKUTEN"]

# Supprime la collection X_train_final
db["X_train_final"].drop()
print("Collection 'X_train_final' supprimée")
db["Y_train_final"].drop()
print("Collection 'Y_train_final' supprimée")

db["X_test_final"].drop()
print("Collection 'X_test_final' supprimée")
db["Y_test_final"].drop()
print("Collection 'Y_test_final' supprimée")