main:
	reverse-proxy-publish apply
	streamlit run test1_revised.py --server.port=57700 --server.address=127.0.0.2
