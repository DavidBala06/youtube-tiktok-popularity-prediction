# Create a test CSV file (test_videos.csv)
import pandas as pd

test_data = {
    'Title': ['TOP 10 AMAZING TRICKS!', 'How to become popular on TikTok', 'MY DAY VLOG'],
    'Published At': ['2024-01-15', '2024-01-16', '2024-01-17'],
    'Keyword': ['tips', 'tutorial', 'vlog']
}

df = pd.DataFrame(test_data)
df.to_csv('test_videos.csv', index=False)