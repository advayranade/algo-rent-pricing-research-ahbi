from sec_edgar_downloader import Downloader

dl = Downloader("Advay Ranade", "advaymranade@gmail.com", "~/code/algorithmic-rent-pricing-ahbi/data/raw/")

ticker = input("Enter the ticker symbol of the REIT you want to pull 10-K filings for: ").strip().upper()

dl.get("10-K", ticker, after="2017-01-01", before="2024-01-01")

