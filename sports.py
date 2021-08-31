import home, football
import streamlit as st
hide_streamlit_style = """
					<style>
					#MainMenu {visibility: hidden;}
					footer {visibility: hidden;}
					</style>
					"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
PAGES = {
    "Home": home,
    "Football": football
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.main()
