import streamlit as st
import plotly.express as px


def bar_plot(df, var):
	var_count = df[var].value_counts().reset_index()
	var_count.columns = [var, "Count"]
	fig_var = px.bar(var_count, x = var, y = "Count",
				  colors = var, title = var)
	st.plotly_chart(fig_var)