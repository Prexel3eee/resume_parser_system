"""Constants for visa types and US states."""

US_VISAS = {
    "h1b": "H-1B Specialty Occupations",
    "h-1b": "H-1B Specialty Occupations",
    "l1": "L-1 Intracompany Transfer",
    "l-1": "L-1 Intracompany Transfer",
    "f1": "F-1 Student Visa",
    "f-1": "F-1 Student Visa",
    "opt": "Optional Practical Training",
    "cpt": "Curricular Practical Training",
    "gc": "Green Card",
    "green card": "Green Card",
    "us citizen": "US Citizen",
    "citizen": "US Citizen",
    "usc": "US Citizen",
    "ead": "Employment Authorization Document",
    "tn": "TN NAFTA Professionals",
    "h4": "H-4 Dependent",
    "h-4": "H-4 Dependent",
    "j1": "J-1 Exchange Visitor",
    "j-1": "J-1 Exchange Visitor",
    "b1": "B-1 Business Visitor",
    "b-1": "B-1 Business Visitor",
    "b2": "B-2 Tourist Visitor",
    "b-2": "B-2 Tourist Visitor",
    "o1": "O-1 Extraordinary Ability",
    "o-1": "O-1 Extraordinary Ability",
    "e3": "E-3 Specialty Occupation (Australia)",
    "e-3": "E-3 Specialty Occupation (Australia)",
    "permanent resident": "Green Card",
    "lawful permanent resident": "Green Card",
    "asylee": "Asylee",
    "refugee": "Refugee"
}

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC"
}

US_STATE_ABBR = {abbr: name.title() for name, abbr in US_STATES.items()}

US_TAX_TERMS = [
    "w2", "w-2", "c2c", "corp to corp", "corp-to-corp", "1099", "contract",
    "full time", "permanent", "c2h", "contract to hire", "hourly", "salary"
] 