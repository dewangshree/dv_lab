import pandas as pd
import matplotlib.pyplot as plt


data={
    'plant_name': ['tomato','lemon','capsicum'],
    'sunlight_exposure': [20, 30, 40 ],
    'plant_height': [67, 89, 12 ]
}
df = pd.DataFrame(data)
df.head()

plt.scatter(df['sunlight_exposure'],df['plant_height'],color="r")
plt.title("sunlight vs plant height")
plt.xlabel("sunlight expo")
plt.ylabel("plant height(cm)")

reduced_df=df[['sunlight_exposure','plant_height']]
reduced_df.corr()

corr_coeff=reduced_df['sunlight_exposure'].corr(df['plant_height'])
print(f"corr coeff :{corr_coeff}")

if corr_coeff <0:
    sign ="negative"
elif corr_coeff >0:
    sign ="positive"
else:
    sign ="neither"
print(f"correlation coeff is{sign}")
strength="strong" if abs(corr_coeff)>0.5 else "weak"
print(f"the coerrelation is {strength}")
if abs(corr_coeff)>0:
    print(f"yes there is{strength} {sign} linear relationship btn sunlight and plant height")
else:
    print(f"no there is linear relationship btn sunlight and palant height")
if strength =="strong":
    print(f"yes there is relationship  betn sunlight expo and plant height")
elif strength == "weak":
    print(f" there is  no sgnificant ass between sunlight expo plant height")
elif strength =="neither":
    print(f"there is no relation ")
