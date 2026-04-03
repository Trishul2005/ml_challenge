import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def clean_numerical(val):
    """Extract first numerical value from string if exists."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return val
    # Try extracting numbers via regex
    match = re.search(r'(\d+)', str(val))
    if match:
        return float(match.group(1))
    return np.nan

def clean_likert(val):
    """Extract the first digit from Likert scale strings like '4 - Agree'."""
    if pd.isna(val) or val == '':
        return np.nan
    s = str(val).strip()
    if s and s[0].isdigit():
        return int(s[0])
    return np.nan

def main():
    csv_path = 'ml_challenge_dataset.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading data from {csv_path}...")
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")

    # Core numerical columns
    emotion_col = "On a scale of 1–10, how intense is the emotion conveyed by the artwork?"
    colors_col = "How many prominent colours do you notice in this painting?"
    objects_col = "How many objects caught your eye in the painting?"
    pay_col = "How much (in Canadian dollars) would you be willing to pay for this painting?"
    
    # Emotion labels (Likert)
    likert_cols = [
        "This art piece makes me feel sombre.",
        "This art piece makes me feel content.",
        "This art piece makes me feel calm.",
        "This art piece makes me feel uneasy."
    ]

    print("Cleaning data columns...")
    # Clean the data
    df['Emotion Intensity'] = pd.to_numeric(df[emotion_col], errors='coerce')
    df['Color Count'] = df[colors_col].apply(clean_numerical)
    df['Object Count'] = df[objects_col].apply(clean_numerical)
    df['Clean Pay'] = df[pay_col].apply(clean_numerical)

    for col in likert_cols:
        if col in df.columns:
            short_name = col.split("feel ")[-1].replace(".", "")
            df[short_name] = df[col].apply(clean_likert)

    # Food Data Processing
    food_col = "If this painting was a food, what would be?"
    df['Food_Clean'] = df[food_col].str.lower().str.strip().str.replace(r'[^\w\s]', '', regex=True)
    
    # Calculate stats per food for the global scatter plot
    food_stats = df.groupby('Food_Clean').agg({
        'Emotion Intensity': ['mean', 'count'],
        'Color Count': 'mean'
    }).reset_index()
    food_stats.columns = ['Food', 'Avg_Emotion', 'Mentions', 'Avg_Colors']
    top_foods = food_stats[food_stats['Mentions'] > 1].sort_values('Mentions', ascending=False)

    # Food per Painting Analysis
    # Get top 7 foods for each painting
    top_food_by_painting = df.groupby(['Painting', 'Food_Clean']).size().reset_index(name='Count')
    top_food_by_painting = top_food_by_painting.sort_values(['Painting', 'Count'], ascending=[True, False])
    top_food_by_painting = top_food_by_painting.groupby('Painting').head(7) 

    # Season Data Processing
    season_col = "What season does this art piece remind you of?"
    # Split seasons if multiple are listed (e.g. "Fall, Winter")
    df_seasons = df.copy()
    df_seasons[season_col] = df_seasons[season_col].str.split(',')
    df_seasons = df_seasons.explode(season_col).reset_index(drop=True)
    df_seasons[season_col] = df_seasons[season_col].str.strip()

    # Filter out extreme outliers for 'Pay' visualization
    df_pay = df[df['Clean Pay'] < 10000].dropna(subset=['Clean Pay'])

    print("Generating visualizations...")
    # 4x2 grid to fit the new Season plot
    fig, axes = plt.subplots(4, 2, figsize=(18, 28))
    plt.subplots_adjust(hspace=0.45, wspace=0.3)

    # 1. Distribution of Emotion Intensity
    sns.histplot(data=df, x='Emotion Intensity', hue='Painting', kde=True, multiple="stack", ax=axes[0, 0])
    axes[0, 0].set_title('1. Emotion Intensity Distribution', fontsize=14)

    # 2. Relationship between Colors and Objects
    sns.scatterplot(data=df, x='Color Count', y='Object Count', hue='Painting', alpha=0.6, ax=axes[0, 1])
    axes[0, 1].set_title('2. Colors vs Objects by Painting', fontsize=14)

    # 3. Mood breakdown
    mood_vars = [col.split("feel ")[-1].replace(".", "") for col in likert_cols if col in df.columns]
    emotion_melt = df.melt(id_vars='Painting', value_vars=mood_vars, 
                           var_name='Mood', value_name='Rating')
    sns.boxplot(data=emotion_melt, x='Mood', y='Rating', hue='Painting', ax=axes[1, 0])
    axes[1, 0].set_title('3. Mood Ratings by Painting', fontsize=14)
    axes[1, 0].set_ylim(0, 6)

    # 4. Global Food Scatter Plot
    sns.scatterplot(data=top_foods.head(25), x='Mentions', y='Avg_Emotion', 
                            size='Avg_Colors', hue='Avg_Emotion', palette='magma', 
                            sizes=(100, 600), ax=axes[1, 1], legend=None)
    
    for i in range(min(12, len(top_foods))):
        axes[1, 1].text(top_foods.iloc[i]['Mentions']+0.3, top_foods.iloc[i]['Avg_Emotion'], 
                      top_foods.iloc[i]['Food'], fontsize=11, weight='bold')
    
    axes[1, 1].set_title('4. Food Association: Overall Frequency vs Emotion', fontsize=14)

    # 5. Willingness to Pay
    sns.kdeplot(data=df_pay, x='Clean Pay', hue='Painting', fill=True, ax=axes[2, 0])
    axes[2, 0].set_title('5. Willingness to Pay ($ < 10k)', fontsize=14)

    # 6. Food In Relation to Painting
    sns.barplot(data=top_food_by_painting, x='Count', y='Food_Clean', hue='Painting', ax=axes[2, 1])
    axes[2, 1].set_title('6. Top Food Mentions per Painting', fontsize=14)
    axes[2, 1].set_xlabel('Count')
    axes[2, 1].set_ylabel('Food Item')
    axes[2, 1].legend(title='Painting', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 7. Seasonal Association
    sns.countplot(data=df_seasons, x=season_col, hue='Painting', order=['Spring', 'Summer', 'Fall', 'Winter'], ax=axes[3, 0])
    axes[3, 0].set_title('7. Seasonal Association per Painting', fontsize=14)
    axes[3, 0].set_xlabel('Season')
    axes[3, 0].set_ylabel('Frequency')

    # 8. Missing Data Visualization (The requested change)
    # Count missing values per row, then group by painting
    df['MissingCount'] = df.isnull().sum(axis=1)
    missing_data = df.groupby('Painting')['MissingCount'].mean().reset_index()
    
    sns.barplot(data=missing_data, x='Painting', y='MissingCount', palette='Reds_r', ax=axes[3, 1])
    axes[3, 1].set_title('8. Avg Missing Responses per Person', fontsize=14)
    axes[3, 1].set_ylabel('Avg Count of Empty Fields')
    axes[3, 1].set_xlabel('')
    
    # Add labels on top of bars
    for i, p in enumerate(missing_data['MissingCount']):
        axes[3, 1].text(i, p + 0.1, f'{p:.2f}', ha='center', fontsize=12)

    plt.suptitle('ML Challenge: Multi-Dimensional Art Analysis', fontsize=26, y=0.99)
    
    # Save the plot
    output_img = 'data_visualization_full_analysis.png'
    plt.savefig(output_img, bbox_inches='tight')
    print(f"Success! Full analysis dashboard saved as {output_img}")
    
    # Show summary statistics
    print("\n--- Season Associations ---")
    print(df_seasons.groupby(['Painting', season_col]).size().unstack(fill_value=0))

    # --- Soundtrack Word Cloud Visualization ---
    try:
        from wordcloud import WordCloud, STOPWORDS
        
        print("Processing soundtracks for word clouds...")
        sound_col = "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."
        
        # Prepare text per painting
        paintings = df['Painting'].unique()
        fig_swc, axes_swc = plt.subplots(1, len(paintings), figsize=(20, 8))
        
        # Musical stopwords
        music_stopwords = set(STOPWORDS)
        music_stopwords.update(['painting', 'soundtrack', 'music', 'sound', 'song', 'rhythm', 'melody', 'track', 'instrumental', 'notes', 'piece', 'instruments'])

        for i, painting in enumerate(paintings):
            text = " ".join(df[df['Painting'] == painting][sound_col].dropna().astype(str))
            
            if text:
                wordcloud = WordCloud(
                    width=800, height=800,
                    background_color='black', # Dark background for soundtracks
                    stopwords=music_stopwords,
                    min_font_size=10,
                    colormap='plasma' # Atmospheric colors
                ).generate(text)

                axes_swc[i].imshow(wordcloud)
                axes_swc[i].set_title(f'Soundtrack Vibes: {painting}', fontsize=16)
                axes_swc[i].axis("off")
            else:
                axes_swc[i].text(0.5, 0.5, 'No data', ha='center')
                axes_swc[i].axis("off")

        plt.tight_layout(pad=0)
        swc_output = 'soundtrack_wordcloud.png'
        plt.savefig(swc_output)
        print(f"Success! Soundtrack word clouds saved as {swc_output}")
        
    except Exception as e:
        print(f"Could not generate soundtrack word cloud: {e}")

if __name__ == "__main__":
    main()

