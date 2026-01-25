from utils.queries import get_facility_mapping_for_user
import json

mapping = get_facility_mapping_for_user({"role": "national"})
names = sorted(list(mapping.keys()))

with open("facility_dump.json", "w") as f:
    json.dump(names, f, indent=4)

print(f"Total facilities: {len(names)}")
print("First 10 names:")
for n in names[:10]:
    print(f"- {n}")

# Check specifically for Chagni
chagni_hits = [n for n in names if "chagni" in n.lower()]
print(f"\nChagni matches: {chagni_hits}")
