import json

with open('data/college_data.json') as f:
    college = json.load(f)
    
print("✅ COLLEGE_DATA.JSON ENHANCEMENTS:")
print("- Keywords in fees section:", "keywords" in college.get("fees", {}))
print("- FAQ entries count:", len(college.get("faq", [])))
print("- Payment modes in engineering:", "payment_modes" in college.get("fees",{}).get("engineering", {}))
print("- Contact info added:", "contact" in college)

with open('data/intents.json') as f:
    intents_data = json.load(f)

tags = [i['tag'] for i in intents_data.get('intents', [])]
print("\n✅ INTENTS.JSON ENHANCEMENTS:")
print("- Total intents:", len(tags))
print("- Has out_of_scope intent:", "out_of_scope" in tags)
print("- Has clarification intent:", "clarification" in tags)
print("- All intents:", tags)
print("- Total patterns (all intents):", sum(len(i.get("patterns", [])) for i in intents_data.get('intents', [])))
