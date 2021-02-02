import streamlit as st



hide_streamlit_style = """
					<style>
					#MainMenu {visibility: hidden;}
					footer {visibility: hidden;}
					</style>
					"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
	st.header("This is an advance sports data explorer.")
	st.write("Please select a page on the left for different analysis.")
	st.markdown("![Alt Text](https://media1.tenor.com/images/13b1dcef2c6b71662e041b78cebab4af/tenor.gif?itemid=15325397)")


if __name__ == "__main__":
    main()