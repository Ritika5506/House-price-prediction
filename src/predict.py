import pickle

model = pickle.load(open("models/house_price_model.pkl", "rb"))

rooms = float(input("Enter number of rooms: "))
lstat = float(input("Enter LSTAT value: "))
ptratio = float(input("Enter PTRATIO value: "))

prediction = model.predict([[rooms, lstat, ptratio]])

print("Predicted House Price:", prediction[0])