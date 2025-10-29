import pandas as pd
import random

goals = ['Strength', 'Flexibility', 'Endurance']
workout_types = ['strength', 'stretching', 'cardio', 'yoga', 'HIIT', 'pilates', 'cycling', 'running', 'swimming']

rows = []
for goal in goals:
    for wt in workout_types:
        for _ in range(50):  # 50 examples per combination
            avg_cal = random.randint(1800, 2800)
            rows.append([goal, avg_cal, wt])

df = pd.DataFrame(rows, columns=['goal', 'avg_calories', 'workout_type'])
df.to_csv('workout_data_large.csv', index=False)
