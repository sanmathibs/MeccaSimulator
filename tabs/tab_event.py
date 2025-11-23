import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from utils.holiday_utils import get_holiday
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go

# --- Load data ---
@st.cache_data
def load_data(file_path="data/all_stores.xlsx", window=10):
    df = pd.read_excel(file_path)
    cols = ["SalesActual", "HoursActual", "SalesForecast", "HoursForecast"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")

    df['Event'] = df['Date'].apply(get_holiday)
    

    event_rows = df[df['Event'].notna()]
    df['EventWindow'] = np.nan
    for idx, row in event_rows.iterrows():
        start = row['Date'] - pd.Timedelta(days=window)
        end = row['Date'] + pd.Timedelta(days=window)
        df.loc[(df['Date'] >= start) & (df['Date'] <= end), 'EventWindow'] = row['Event']
    
    return df
# --- Event Filter ---
def select_stores_and_events(df):

    col1, col2 = st.columns(2) 

    # --- Event multi-select ---
    with col1:
        event_options = df['Event'].dropna().unique()
        selected_events = st.multiselect(
            "Select Event Types",
            event_options,
            default=[]  
        )
        if not selected_events:  
            selected_events = list(event_options)

    # --- Store multi-select ---
    with col2:
        store_options = df['Store'].dropna().unique()
        selected_stores = st.multiselect(
            "Select Stores",
            store_options,
            default=[]  
        )
        if not selected_stores:  
            selected_stores = list(store_options)

    return selected_events, selected_stores


def compute_event_uplift(df_event):
    """
    (SalesActual - SalesForecast) / SalesForecast * 100
    """
    df_event = df_event.copy()
    df_event['Uplift'] = ((df_event['SalesActual'] - df_event['SalesForecast']) / 
                      df_event['SalesForecast'].replace(0, np.nan) * 100)
    df_event = df_event.dropna(subset=['Uplift'])

    return df_event

def render_event_calendar_heatmap(df, selected_events=[], selected_stores=[]):
    df_filtered = df.copy()
    if selected_events:
        df_filtered = df_filtered[df_filtered['EventWindow'].isin(selected_events)]
    if selected_stores:
        df_filtered = df_filtered[df_filtered['Store'].isin(selected_stores)]

    if df_filtered.empty:
        st.info("No data for selected stores/events")
        return

    # calculate uplift
    df_filtered = compute_event_uplift(df_filtered)

    # Pivot 
    pivot_table = df_filtered.pivot_table(
        index='Store',
        columns='EventWindow',
        values='Uplift',
        aggfunc='mean'
    )

    # Heatmap
    fig = px.imshow(
        pivot_table,
        color_continuous_scale='RdYlGn',
        labels=dict(x="Event", y="Store", color="Uplift"),
        text_auto=".2f",
        aspect="auto",
        title="Event with Uplift"
    )

    fig.update_layout(
        yaxis={'categoryorder':'category ascending'},
        xaxis_title="",
        yaxis_title="",
        title_x=0.5,
    )

    st.plotly_chart(fig, width='stretch')


def render_event_uplift_analysis(df, selected_stores=[], selected_events=[], window=10):
    """
    Event Uplift Analysis
    - Bar chart: average uplift per event type vs normal day
    - Table: sales, uplift %, forecast accuracy
    """
    if not selected_stores:
        st.info("Select at least one store for analysis")
        return
    
    # --- Filter data ---
    df_filtered = df[df['Store'].isin(selected_stores)]
    if selected_events:
        df_filtered = df_filtered[df_filtered['EventWindow'].isin(selected_events)]

    if df_filtered.empty:
        st.info("No data for selected stores/events")
        return

    # --- Calculate uplift for each row ---
    df_filtered = compute_event_uplift(df_filtered)

   
    uplift_summary = []
    for event in df_filtered['EventWindow'].dropna().unique():
        df_event = df_filtered[df_filtered['EventWindow'] == event]
        avg_uplift = df_event['Uplift'].mean()
        avg_sales = df_event['SalesActual'].mean()
        avg_sale_forecast = df_event['SalesForecast'].mean()
        forecast_accuracy = 100 * (1 - abs(df_event['SalesActual'] - df_event['SalesForecast']) / 
                                   df_event['SalesForecast'].replace(0, np.nan)).mean()
        uplift_summary.append({
            "Event": event,
            "Avg Sales": avg_sales,
            "Avg Forecast Sales": avg_sale_forecast,
            "Uplift %": avg_uplift,
            "Forecast Accuracy %": forecast_accuracy
            
        })
    df_summary = pd.DataFrame(uplift_summary).sort_values("Event")

    # --- Bar chart: average uplift per event ---
    fig = px.bar(
    df_summary,
    x="Event",
    y="Uplift %",
    color="Uplift %",
    text=df_summary["Uplift %"],
    title="Average Uplift per Event",
    color_continuous_scale="RdYlGn",
)


    fig.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='inside',
        insidetextanchor='middle'
    )

   
    avg_uplift = df_summary.groupby('Event')['Uplift %'].mean().reset_index()


    fig.add_trace(
        go.Scatter(
            x=avg_uplift['Event'],
            y=avg_uplift['Uplift %'],
            mode='lines+markers',
            name='Average',
            line=dict(color='black', width=1),
            marker=dict(size=5)
        )
    )

    fig.update_layout(title_x=0.5)

    st.plotly_chart(fig, width='stretch')

    # --- Table ---
    st.subheader("Detailed Event Uplift Table")
    st.dataframe(df_summary.style.format({
        "Avg Sales": "{:,.0f}",
        "Avg Forecast Sales": "{:,.0f}",
        "Uplift %": "{:.2f}",
        "Forecast Accuracy %": "{:.2f}"
        
    }))
    
    return df_summary


def render_store_level_table(df, selected_stores=[], selected_events=[]):

    df_filtered = df.copy()
    if selected_stores:
        df_filtered = df_filtered[df_filtered['Store'].isin(selected_stores)]
    if selected_events:
        df_filtered = df_filtered[df_filtered['EventWindow'].isin(selected_events)]

    if df_filtered.empty:
        st.info("No data for selected stores/events")
        return

    df_filtered['Uplift'] = ((df_filtered['SalesActual'] - df_filtered['SalesForecast']) / 
                             df_filtered['SalesForecast'].replace(0, np.nan) * 100)
    df_filtered = df_filtered.dropna(subset=['Uplift'])


    stores_to_show = selected_stores if selected_stores else df_filtered['Store'].unique()
    n_store = len(stores_to_show)


    cols = st.columns(n_store)

    for i, store in enumerate(stores_to_show):
        df_store = df_filtered[df_filtered['Store'] == store]
        if df_store.empty:
            continue


        agg_list = []
        for ev in df_store['EventWindow'].dropna().unique():
            df_ev_store = df_store[df_store['EventWindow'] == ev]
            uplift = df_ev_store['Uplift'].mean()
            avg_sales = df_ev_store['SalesActual'].mean()
            avg_forecast = df_ev_store['SalesForecast'].mean()
            forecast_acc = 100 * (1 - abs(df_ev_store['SalesActual'] - df_ev_store['SalesForecast']) / 
                                  df_ev_store['SalesForecast'].replace(0, np.nan)).mean()
            agg_list.append({
                "Event": ev,
                "Uplift %": uplift,
                "Avg Sales": avg_sales,
                "Avg Forecast Sales": avg_forecast,
                "Forecast Accuracy %": forecast_acc
            })

        df_store_summary = pd.DataFrame(agg_list).sort_values('Event').reset_index(drop=True)

        with cols[i]:
            st.subheader(f"{store}")
            st.dataframe(
                df_store_summary.style.format({
                    "Avg Sales": "{:,.0f}",
                    "Avg Forecast Sales": "{:,.0f}",
                    "Uplift %": "{:.2f}",
                    "Forecast Accuracy %": "{:.2f}"
                })
            )

# --- Main render ---
def render():
    df = load_data()
    st.header("Event Analysis")
    selected_events,select_store = select_stores_and_events(df)
    render_event_calendar_heatmap(df, selected_events=selected_events, selected_stores=select_store)
    render_event_uplift_analysis(df, selected_stores=select_store, selected_events=selected_events)
    render_store_level_table(df, selected_stores=select_store, selected_events=selected_events)
    