import random
import pandas as pd

# =========================
# Templates
# =========================

english_templates = {
    "genuine": [
        "Your OTP for login is {otp}. Do not share it with anyone.",
        "Dear customer, your transaction of INR {amount} has been processed successfully.",
        "Welcome to our service! Thank you for registering.",
        "Reminder: Your bill payment of INR {amount} is due on {date}.",
        "Hi {name}, your order #{order} has been shipped."
    ],
    "suspected": [
        "Dear customer, get {discount}% off if you verify OTP at {link}",
        "Urgent! Confirm your bank details immediately at {link} to avoid suspension.",
        "You have won INR {amount}! Claim now at {link}.",
        "Your account may be blocked. Verify identity at {link}.",
        "Limited time deal! Recharge and get {bonus}% cashback."
    ],
    "confirmed_spam": [
        "Congratulations! You won INR {amount} lottery. Share OTP {otp} at {link}.",
        "Your bank account will be blocked in 24 hours. Visit {link} now.",
        "Final warning! Update PAN card details here: {link}.",
        "Dear user, send OTP {otp} to claim free gift worth INR {amount}.",
        "Click {link} now to unlock your frozen account."
    ]
}

hinglish_templates = {
    "genuine": [
        "Dear Customer, aapka OTP hai {otp}. Kripya isse kisi ke saath share na karein.",
        "Namaste {name}, INR {amount} ka transaction successful ho gaya hai.",
        "Swagat hai! Aapka registration safal raha.",
        "Yaad rahe, aapka bill INR {amount} ka {date} tak due hai.",
        "Hi {name}, aapka order #{order} dispatch ho gaya hai."
    ],
    "suspected": [
        "Dear customer, turant verify karein apna account {link} par.",
        "Jaldi karein! Aapka account block ho sakta hai. Visit {link}.",
        "Aap jeet gaye ho INR {amount}! Claim karne ke liye click karein {link}.",
        "Recharge par {bonus}% cashback! Sirf aaj ke liye.",
        "OTP {otp} dal kar apna reward paayein. Link: {link}"
    ],
    "confirmed_spam": [
        "Badhai ho! Aap INR {amount} ke lottery jeet gaye. OTP {otp} daalein aur {link} visit karein.",
        "Aapka bank account block ho gaya hai. Turant {link} par login karein.",
        "Kripya PAN details update karein {link} par, warna account band ho jayega.",
        "Free gift worth INR {amount} paane ke liye OTP {otp} bhejein.",
        "24 ghante me account suspend ho jayega. Abhi {link} par click karein."
    ]
}

# =========================
# Helper function
# =========================
def generate_message(template, lang):
    otp = random.randint(100000, 999999)
    amount = random.randint(1000, 50000)
    discount = random.choice([10, 20, 30, 50])
    bonus = random.choice([5, 10, 20, 50])
    date = f"{random.randint(1,28)}/0{random.randint(1,9)}/2025"
    name = random.choice(["Amit", "Riya", "Suresh", "Neha", "Arjun"])
    order = random.randint(10000, 99999)
    link = random.choice([
        "http://secure-login.com",
        "http://bank-update.net",
        "http://fraudlink.com",
        "http://verify-now.org"
    ])
    
    msg = template.format(otp=otp, amount=amount, discount=discount,
                          bonus=bonus, date=date, name=name, order=order, link=link)
    return msg

# =========================
# Dataset Generation
# =========================
def generate_dataset(n_per_class=2000):
    data = []
    for lang, templates in [("english", english_templates), ("hinglish", hinglish_templates)]:
        for label, msgs in templates.items():
            for _ in range(n_per_class):
                template = random.choice(msgs)
                text = generate_message(template, lang)
                data.append({"text": text, "label": label, "language": lang})
    return pd.DataFrame(data)

# Generate dataset
df = generate_dataset(2000)

# Save to CSV
df.to_csv("sms_fraud_dataset.csv", index=False)

print("Dataset generated successfully with shape:", df.shape)
print(df.sample(10))
