import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tariff_predictor import FashionTariffPredictor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TariffDataAnalyzer:
    def __init__(self):
        self.predictor = FashionTariffPredictor()
        
    def generate_analysis_data(self, n_samples=2000):
        """Generate comprehensive data for analysis"""
        return self.predictor.generate_sample_data(n_samples)
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations of tariff data"""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Fashion Tariff Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Tariff by Category
        category_tariff = df.groupby('category')['tariff_amount'].mean().sort_values(ascending=False)
        axes[0, 0].bar(category_tariff.index, category_tariff.values, color='skyblue')
        axes[0, 0].set_title('Average Tariff by Category')
        axes[0, 0].set_ylabel('Average Tariff ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Tariff Rate Distribution
        axes[0, 1].hist(df['tariff_rate'] * 100, bins=30, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Distribution of Tariff Rates')
        axes[0, 1].set_xlabel('Tariff Rate (%)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Value vs Tariff Scatter
        scatter = axes[0, 2].scatter(df['value_usd'], df['tariff_amount'], 
                                   c=df['category'].astype('category').cat.codes, 
                                   alpha=0.6, cmap='tab10')
        axes[0, 2].set_title('Item Value vs Tariff Amount')
        axes[0, 2].set_xlabel('Item Value ($)')
        axes[0, 2].set_ylabel('Tariff Amount ($)')
        
        # 4. Tariff by Origin Country
        country_tariff = df.groupby('origin_country')['tariff_rate'].mean().sort_values(ascending=False)
        axes[1, 0].bar(country_tariff.index, country_tariff.values * 100, color='coral')
        axes[1, 0].set_title('Average Tariff Rate by Origin Country')
        axes[1, 0].set_ylabel('Average Tariff Rate (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Material Impact
        material_tariff = df.groupby('material')['tariff_amount'].mean().sort_values(ascending=False)
        axes[1, 1].barh(material_tariff.index, material_tariff.values, color='gold')
        axes[1, 1].set_title('Average Tariff by Material')
        axes[1, 1].set_xlabel('Average Tariff ($)')
        
        # 6. Brand Tier Analysis
        brand_stats = df.groupby('brand_tier').agg({
            'tariff_rate': 'mean',
            'value_usd': 'mean'
        }).round(3)
        
        x = np.arange(len(brand_stats.index))
        width = 0.35
        
        ax2 = axes[1, 2].twinx()
        bars1 = axes[1, 2].bar(x - width/2, brand_stats['tariff_rate'] * 100, 
                              width, label='Avg Tariff Rate (%)', color='purple', alpha=0.7)
        bars2 = ax2.bar(x + width/2, brand_stats['value_usd'], 
                       width, label='Avg Value ($)', color='orange', alpha=0.7)
        
        axes[1, 2].set_title('Brand Tier Analysis')
        axes[1, 2].set_xlabel('Brand Tier')
        axes[1, 2].set_ylabel('Tariff Rate (%)', color='purple')
        ax2.set_ylabel('Average Value ($)', color='orange')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(brand_stats.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig('tariff_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_interactive_dashboard(self, df):
        """Create interactive Plotly dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Tariff by Category', 'Value vs Tariff', 
                          'Country Analysis', 'Material Impact'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Category analysis
        category_data = df.groupby('category')['tariff_amount'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=category_data.index, y=category_data.values, 
                   name='Avg Tariff', marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. Value vs Tariff scatter
        fig.add_trace(
            go.Scatter(x=df['value_usd'], y=df['tariff_amount'],
                      mode='markers', name='Items',
                      marker=dict(color=df['category'].astype('category').cat.codes,
                                colorscale='viridis', opacity=0.6),
                      text=df['category']),
            row=1, col=2
        )
        
        # 3. Country analysis
        country_data = df.groupby('origin_country')['tariff_rate'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=country_data.index, y=country_data.values * 100,
                   name='Tariff Rate %', marker_color='coral'),
            row=2, col=1
        )
        
        # 4. Material analysis
        material_data = df.groupby('material')['tariff_amount'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=material_data.values, y=material_data.index,
                   orientation='h', name='Avg Tariff', marker_color='gold'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Interactive Tariff Analysis Dashboard")
        fig.write_html('interactive_dashboard.html')
        
        return fig
    
    def generate_insights(self, df):
        """Generate key insights from the data"""
        insights = []
        
        # Category insights
        category_stats = df.groupby('category').agg({
            'tariff_amount': ['mean', 'std'],
            'tariff_rate': 'mean',
            'value_usd': 'mean'
        }).round(2)
        
        highest_tariff_category = category_stats[('tariff_amount', 'mean')].idxmax()
        lowest_tariff_category = category_stats[('tariff_amount', 'mean')].idxmin()
        
        insights.append(f"📊 **Category Analysis:**")
        insights.append(f"   • Highest tariff category: {highest_tariff_category}")
        insights.append(f"   • Lowest tariff category: {lowest_tariff_category}")
        
        # Country insights
        country_stats = df.groupby('origin_country')['tariff_rate'].mean().sort_values(ascending=False)
        highest_tariff_country = country_stats.index[0]
        lowest_tariff_country = country_stats.index[-1]
        
        insights.append(f"\n🌍 **Country Analysis:**")
        insights.append(f"   • Highest tariff origin: {highest_tariff_country} ({country_stats.iloc[0]*100:.1f}%)")
        insights.append(f"   • Lowest tariff origin: {lowest_tariff_country} ({country_stats.iloc[-1]*100:.1f}%)")
        
        # Material insights
        material_stats = df.groupby('material')['tariff_amount'].mean().sort_values(ascending=False)
        expensive_material = material_stats.index[0]
        cheap_material = material_stats.index[-1]
        
        insights.append(f"\n🧵 **Material Analysis:**")
        insights.append(f"   • Most expensive material (tariff): {expensive_material}")
        insights.append(f"   • Least expensive material (tariff): {cheap_material}")
        
        # Value insights
        value_correlation = df['value_usd'].corr(df['tariff_amount'])
        insights.append(f"\n💰 **Value Analysis:**")
        insights.append(f"   • Correlation between value and tariff: {value_correlation:.3f}")
        
        # Brand tier insights
        brand_stats = df.groupby('brand_tier')['tariff_rate'].mean().sort_values(ascending=False)
        insights.append(f"\n⭐ **Brand Tier Analysis:**")
        for tier, rate in brand_stats.items():
            insights.append(f"   • {tier}: {rate*100:.1f}% average tariff rate")
        
        return "\n".join(insights)

def main():
    """Run comprehensive analysis"""
    analyzer = TariffDataAnalyzer()
    
    print("Generating analysis data...")
    df = analyzer.generate_analysis_data(2000)
    
    print("Creating visualizations...")
    analyzer.create_visualizations(df)
    
    print("Creating interactive dashboard...")
    analyzer.create_interactive_dashboard(df)
    
    print("Generating insights...")
    insights = analyzer.generate_insights(df)
    print("\n" + "="*50)
    print("KEY INSIGHTS")
    print("="*50)
    print(insights)
    
    # Save insights to file
    with open('tariff_insights.txt', 'w') as f:
        f.write(insights)
    
    print(f"\nAnalysis complete! Files generated:")
    print("• tariff_analysis.png - Static visualizations")
    print("• interactive_dashboard.html - Interactive dashboard")
    print("• tariff_insights.txt - Key insights")

if __name__ == "__main__":
    main()
