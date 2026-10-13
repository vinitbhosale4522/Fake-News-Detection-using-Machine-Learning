import os
import pandas as pd
import numpy as np

# Define target dataset directory
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

fake_file = os.path.join(DATASET_DIR, "Fake.csv")
true_file = os.path.join(DATASET_DIR, "True.csv")

# Templates for Real (True) News
real_titles = [
    "Senate Passes Bipartisan Infrastructure Bill After Weeks of Debate",
    "Federal Reserve Keeps Interest Rates Steady Amid Economic Indicators",
    "NASA Launches James Webb Space Telescope New Scientific Deep Space Mission",
    "World Health Organization Releases Global Health Security Guidelines",
    "European Union Finalizes AI Act Framework to Regulate Artificial Intelligence",
    "United Nations Security Council Holds Emergency Session on International Climate Policy",
    "Tech Giants Report Q4 Financial Growth and Cloud Computing Revenue Boost",
    "Scientists Discover New Exoplanet in Habitable Zone of Nearby Star System",
    "Global Trade Agreement Signed by Major Exporting Nations in Geneva",
    "Supreme Court Hears Landmark Case on Digital Privacy and Data Encryption",
    "Electric Vehicle Adoption Reaches Record High in Global Automotive Market",
    "Renewable Energy Investment Overtakes Fossil Fuels for First Time in History",
    "Researchers Publish Breakthrough Treatment Trial Results in Peer-Reviewed Journal",
    "Department of Energy Announces Funding for Next-Generation Solar Technologies",
    "Global Summit Concludes with Commitment to Reduce Methane Emissions",
    "Central Bank Digital Currency Pilot Program Launched by International Financial Consortium",
    "Cybersecurity Cybersecurity Agency Issues Advisory on Cloud Vulnerabilities",
    "Agriculture Department Reports Record Crop Yield Forecasts for Upcoming Season",
    "Bipartisan Coalition Introduces Bill to Enhance National Power Grid Security",
    "International Space Station Astronauts Complete Successful Spacewalk for Solar Upgrade",
    "Labor Department Reports Drop in Unemployment Claims to Multi-Year Low",
    "Treasury Department Announces New Anti-Money Laundering Enforcement Rules",
    "Global Shipping Rates Stabilize Following Resolution of Supply Chain Bottlenecks",
    "Medical Researchers Identify Key Biomarker for Early Stage Disease Detection",
    "National Science Foundation Grants $50 Million for Supercomputing Research Facilities"
]

real_text_templates = [
    "WASHINGTON (Reuters) - Official representatives confirmed today that the legislation was approved following extensive negotiations between committee members. The policy aims to allocate federal funds toward modernizing transportation infrastructure, upgrading public transit networks, and expanding high-speed broadband access across rural communities.",
    "NEW YORK (Reuters) - The Federal Reserve announced its policy decision to maintain current benchmark interest rates after reviewing key inflation metrics and labor market statistics. Economic analysts noted that the central bank remains focused on achieving sustainable price stability while encouraging job growth.",
    "CAPE CANAVERAL (Reuters) - Mission controllers at NASA confirmed successful telemetry signals from the newly launched space observatory. Engineers reported all solar arrays and communication antennas deployed according to schedule as the spacecraft journeys toward its final orbital destination.",
    "GENEVA (Reuters) - The World Health Organization issued comprehensive health recommendations today, urging member nations to strengthen public health infrastructure and surveillance systems. International delegates emphasized the importance of equitable access to medical supplies and global cooperation.",
    "BRUSSELS (Reuters) - European lawmakers voted in favor of landmark regulatory legislation designed to ensure ethical standard enforcement across artificial intelligence systems. The regulations establish strict risk classifications, mandatory transparency requirements, and heavy penalties for non-compliance.",
    "LONDON (Reuters) - Economic indicators released by the financial ministry show steady GDP growth in the last quarter, driven by robust consumer spending and increased manufacturing productivity. Analysts expect economic momentum to continue into the next fiscal year.",
    "TOKYO (Reuters) - Renewable energy capacity expanded rapidly over the past twelve months, according to official energy sector data released today. Solar and wind installations generated over forty percent of total electricity consumption nationwide.",
    "PARIS (Reuters) - Representatives from over fifty countries gathered today to ratify an international treaty aimed at preserving oceanic biodiversity. The historic agreement establishes protected maritime zones and imposes strict limits on deep-sea industrial mining."
]

real_subjects = ["politicsNews", "worldnews"]

# Templates for Fake News
fake_titles = [
    "SHOCKING: Secret Government Hologram Project Exposed by Anonymous Whistleblower!",
    "BOMBSHELL: Hidden Alien Artifact Discovered Beneath Antarctic Ice Sheet!",
    "PROOF: Mind Control Chips Transmitted Through 5G Towers, Insiders Claim!",
    "BREAKING: Secret Underground Tunnel Network Found Below Capital City!",
    "EXPOSED: Celebrity Admits to Using Magic Elixir to Stop Aging Process!",
    "MUST WATCH: Miracle Juice Cures All Known Illnesses in 24 Hours, Doctors Furious!",
    "LEAKED DOCUMENTS: Global Elite Planning to Ban Private Property by Next Month!",
    "UNBELIEVABLE: Man Discovers Infinite Energy Device in Garage Using Magnets!",
    "SHOCK REPORT: Weather Control Machine Caused Yesterday's Heavy Rainfall!",
    "SECRET REVEALED: Time Traveler from 2085 Warns of Impending Catastrophe!",
    "BOMBSHELL: Government Secretly Swapping Real Money with Counterfeit Digital Currency!",
    "PROOF: Ancient Pyramids Were Secretly Power Plants Built by Extra-Terrestrials!",
    "EXPOSED: Mainstream Media Caught Using CGI Actors in Live Evening News Broadcast!",
    "WARNING: Common Tap Water Contains Chemical designed to Control Thoughts!",
    "SHOCKING TRUTH: NASA Faked Mars Rover Footage in Desert Film Studio!",
    "LEAKED VIDEO: Secret Society Performs Ritual in Subterranean Bunker!",
    "BOMBSHELL: Scientists Create Mutant Hybrid Animal in Hidden Offshore Laboratory!",
    "MUST SEE: Mysterious Cloud Formation Above City Hides Alien Spaceship!",
    "EXPOSED: Billionaire Buys Entire Island to Build Secret Underground Bunker Fortress!",
    "PROOF: Ancient Map Shows Secret Hidden Continent Beyond South Pole Ice Wall!"
]

fake_text_templates = [
    "You won't believe what mainstream media is refusing to report! An anonymous whistleblower has come forward with undeniable proof revealing a deep state agenda operating in total secrecy. Top government insiders have reportedly been hiding the truth from the public for decades, using advanced psychological operations to keep citizens distracted while secret funds are diverted into illegal black budget projects.",
    "In a shocking turn of events that the fake news media wants to censor, leaked documents have exposed a massive globalist conspiracy. Insiders claim that shadowy figures are meeting behind closed doors to orchestrate events and control world affairs. Viral social media posts reveal unbelievable footage that the government is trying to scrub from the internet!",
    "BREAKING NEWS: Shocking video proof has emerged on private messaging forums exposing what scientists and corrupt politicians have been hiding! Independent researchers claim that miraculous breakthroughs are being suppressed by greedy corporations who don't want you to know the real truth. Share this story before it gets taken down by big tech censors!",
    "A massive bombshell revelation has just shaken the internet! Anonymous sources close to top intelligence officials report that secret underground facilities are operating high-tech experiments away from public oversight. Citizens across the nation are demanding answers as whistleblowers continue leaking classified files exposing the shocking agenda!"
]

fake_subjects = ["News", "politics", "left-news", "Government News", "US_News", "Middle-east"]

def generate_samples(count=1200):
    # Generate True news records
    true_records = []
    np.random.seed(42)
    for i in range(count):
        t = np.random.choice(real_titles) + f" (Report #{i+1})"
        b = np.random.choice(real_text_templates) + f" Official report ID #{1000+i} stated that key stakeholders affirmed the policy objectives during the annual press briefing."
        s = np.random.choice(real_subjects)
        d = f"December {1 + (i % 28)}, 2017"
        true_records.append({"title": t, "text": b, "subject": s, "date": d})
        
    # Generate Fake news records
    fake_records = []
    for i in range(count):
        t = np.random.choice(fake_titles) + f" - SEE THIS! #{i+1}"
        b = np.random.choice(fake_text_templates) + f" Wake up people! Share this urgent report #{2000+i} with everyone before censors remove the video link from social media!"
        s = np.random.choice(fake_subjects)
        d = f"December {1 + (i % 28)}, 2017"
        fake_records.append({"title": t, "text": b, "subject": s, "date": d})
        
    df_true = pd.DataFrame(true_records)
    df_fake = pd.DataFrame(fake_records)
    
    df_true.to_csv(true_file, index=False)
    df_fake.to_csv(fake_file, index=False)
    print(f"Generated {len(df_true)} True news articles in {true_file}")
    print(f"Generated {len(df_fake)} Fake news articles in {fake_file}")

if __name__ == "__main__":
    generate_samples(count=1500)
