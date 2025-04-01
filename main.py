import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

# Configuração da página
st.set_page_config(page_title="People Analytics", layout="wide")

# Menu lateral
nav = get_nav_from_toml(".streamlit/pages.toml")
pg = st.navigation(nav)

add_page_title(pg)
pg.run()