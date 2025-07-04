#!/usr/bin/env python3
"""
Streamlit dashboard for visualizing benchmark results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Gemma Benchmark Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_benchmark_results(results_path="benchmarks_output"):
    """Load benchmark results from the output directory."""
    results = {}
    
    # Look for JSON files in the results directory
    results_dir = Path(results_path)
    if results_dir.exists():
        for json_file in results_dir.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results[json_file.stem] = data
            except Exception as e:
                st.error(f"Error loading {json_file}: {e}")
    
    return results

def create_performance_heatmap(results_data):
    """Create a performance heatmap using Plotly."""
    if not results_data:
        return None
    
    # Extract data for heatmap
    models = []
    tasks = []
    scores = []
    
    for model_name, model_results in results_data.items():
        for task_name, task_results in model_results.items():
            if isinstance(task_results, dict) and "overall" in task_results:
                overall = task_results["overall"]
                if "accuracy" in overall:
                    models.append(model_name)
                    tasks.append(task_name.upper())
                    scores.append(overall["accuracy"])
                elif "pass_at_1" in overall:
                    models.append(model_name)
                    tasks.append(task_name.upper())
                    scores.append(overall["pass_at_1"])
    
    if not scores:
        return None
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Task': tasks,
        'Score': scores
    })
    
    # Pivot for heatmap
    pivot_df = df.pivot(index='Model', columns='Task', values='Score')
    
    # Create heatmap
    fig = px.imshow(
        pivot_df,
        color_continuous_scale='RdYlGn',
        aspect='auto',
        title='Model Performance Heatmap'
    )
    
    fig.update_layout(
        xaxis_title='Tasks',
        yaxis_title='Models',
        height=500
    )
    
    return fig

def create_model_comparison(results_data, selected_task):
    """Create a model comparison chart for a specific task."""
    if not results_data:
        return None
    
    models = []
    scores = []
    
    for model_name, model_results in results_data.items():
        if selected_task in model_results:
            task_results = model_results[selected_task]
            if isinstance(task_results, dict) and "overall" in task_results:
                overall = task_results["overall"]
                if "accuracy" in overall:
                    models.append(model_name)
                    scores.append(overall["accuracy"])
                elif "pass_at_1" in overall:
                    models.append(model_name)
                    scores.append(overall["pass_at_1"])
    
    if not scores:
        return None
    
    # Create bar chart
    fig = px.bar(
        x=models,
        y=scores,
        title=f'Model Performance on {selected_task.upper()}',
        labels={'x': 'Models', 'y': 'Score'}
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_overall_ranking(results_data):
    """Create an overall model ranking chart."""
    if not results_data:
        return None
    
    model_scores = {}
    
    for model_name, model_results in results_data.items():
        scores = []
        for task_name, task_results in model_results.items():
            if isinstance(task_results, dict) and "overall" in task_results:
                overall = task_results["overall"]
                if "accuracy" in overall:
                    scores.append(overall["accuracy"])
                elif "pass_at_1" in overall:
                    scores.append(overall["pass_at_1"])
        
        if scores:
            model_scores[model_name] = np.mean(scores)
    
    if not model_scores:
        return None
    
    # Sort by performance
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    models, scores = zip(*sorted_models)
    
    # Create horizontal bar chart
    fig = px.bar(
        x=scores,
        y=models,
        orientation='h',
        title='Overall Model Ranking',
        labels={'x': 'Average Score', 'y': 'Models'}
    )
    
    fig.update_layout(height=400)
    
    return fig

def main():
    """Main dashboard function."""
    
    # Header
    st.title("🚀 Gemma Benchmark Dashboard")
    st.markdown("Interactive visualization of language model benchmark results")
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # Load results
    results = load_benchmark_results()
    
    if not results:
        st.warning("No benchmark results found. Please run a benchmark first.")
        st.info("Results should be saved in the `benchmarks_output/` directory.")
        return
    
    # Select results file
    result_files = list(results.keys())
    selected_file = st.sidebar.selectbox(
        "Select Results File",
        result_files,
        index=0 if result_files else None
    )
    
    if selected_file:
        results_data = results[selected_file]
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Performance Overview")
            
            # Performance heatmap
            heatmap_fig = create_performance_heatmap(results_data)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            else:
                st.info("No performance data available for heatmap.")
        
        with col2:
            st.subheader("🏆 Overall Ranking")
            
            # Overall ranking
            ranking_fig = create_overall_ranking(results_data)
            if ranking_fig:
                st.plotly_chart(ranking_fig, use_container_width=True)
            else:
                st.info("No ranking data available.")
        
        # Task-specific comparison
        st.subheader("📊 Task-Specific Analysis")
        
        # Get available tasks
        available_tasks = set()
        for model_results in results_data.values():
            available_tasks.update(model_results.keys())
        
        if available_tasks:
            selected_task = st.selectbox(
                "Select Task for Detailed Comparison",
                sorted(available_tasks)
            )
            
            task_fig = create_model_comparison(results_data, selected_task)
            if task_fig:
                st.plotly_chart(task_fig, use_container_width=True)
            else:
                st.info(f"No data available for {selected_task}.")
        
        # Raw results table
        st.subheader("📋 Raw Results")
        
        # Convert to DataFrame for display
        table_data = []
        for model_name, model_results in results_data.items():
            for task_name, task_results in model_results.items():
                if isinstance(task_results, dict) and "overall" in task_results:
                    overall = task_results["overall"]
                    row = {
                        "Model": model_name,
                        "Task": task_name.upper(),
                        "Score": None,
                        "Metric": None
                    }
                    
                    if "accuracy" in overall:
                        row["Score"] = f"{overall['accuracy']:.3f}"
                        row["Metric"] = "Accuracy"
                    elif "pass_at_1" in overall:
                        row["Score"] = f"{overall['pass_at_1']:.3f}"
                        row["Metric"] = "Pass@1"
                    
                    table_data.append(row)
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No results data available for table display.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Dashboard Features:** "
        "• Performance Heatmap • Model Ranking • Task Comparison • Raw Results"
    )

if __name__ == "__main__":
    main() 