# PyroFinder Live Ops — Operational Context

Status: initial draft  
Runtime role: optional context for Live Ops incident reasoning and concise first-message generation.  
Do not use this file for YOLO training or for changing image-zone metadata.

## How this file should be used

Use this Markdown file as human-readable / LLM-readable context.

Use the companion JSON file as the machine-readable source for deterministic code:

```text
data/live_ops/thunder_valley_operational_context.json
```

The existing `live_ops_camera.json` remains the source of truth for camera coordinates, reference points, image polygons, and zone reference points. This file adds operational meaning around the scene: nearby landmarks, sensitive receptors, contact policy, and first-message rules.

## Camera and view context

- Camera name: `ThunderValleyWest`
- Camera host: Thunder Valley Casino Resort
- User-provided mounting context: on the roof of Thunder Valley Casino Resort.
- User-provided viewing direction: north toward Lincoln Crossing.
- Configured camera coordinates from `live_ops_camera.json`: `38.840335705966275, -121.3152325466259`
- Configured camera height: `70 m`

The uploaded camera setup file already includes camera data, seven reference points, three image zones, and `zone_ref_point` values for the current zones. Do not duplicate that geometry here.

## Primary operational places

### Thunder Valley Casino Resort

Role: camera host / populated resort area.  
Official address: `1200 Athens Avenue, Lincoln, CA 95648`.  
Official toll-free number: `877-468-8777`.

Operational relevance:
- A confirmed fire/smoke incident moving toward the resort, venue, parking, or guest areas should be treated as high sensitivity.
- Do not claim casino operating hours unless verified from an official current source.

### Lincoln Crossing

Role: residential community north / northeast in the camera context.  
Public context: the community association describes Lincoln Crossing as located in Lincoln, CA, encompassing over 40 acres and close to 3000 single-family homes.

Operational relevance:
- Smoke/fire inside, near, or moving toward Lincoln Crossing should usually raise urgency.
- The assistant may recommend notifying the relevant local authority or local emergency contact.
- Do not invent HOA/community contacts. Ask the user whether to search/verify a contact if needed.

### Farm owner site

User-provided location:

```text
38.843771, -121.317149
Placer County, California, USA
https://maps.app.goo.gl/h2P76mNrkSe8h3yG8
```

Operational relevance:
- Treat as the likely property/farm owner reference site for the demo.
- If smoke/fire is close to this site or moving toward it, the assistant may suggest preparing a property-owner/site-operator update.
- Contact is unknown. Do not invent owner name, phone, or email.

### Wetland / slough / ponds

Source: user-provided satellite screenshot.  
The screenshot shows wetland/water features northwest of the farm-owner site and a visible map label for `Ingram Slough`.

Operational relevance:
- A detection projected over water/wetland may be lower urgency than a detection near homes or structures.
- Do not dismiss it automatically: smoke movement toward roads, homes, the farm site, casino, or other structures can still make it important.

### Industrial Avenue / Twelve Bridges Drive corridor

Role: road/access corridor visible in the broader map context.  
Operational relevance:
- If smoke threatens road visibility, access, or evacuation routes, recommend notifying relevant local authority.
- Do not overstate road closure or traffic impact unless current live data is available.

### Twelve Bridges High School

Official public details:
- Name: Twelve Bridges High School
- Address: `2360 Fieldstone Drive, Lincoln, CA 95648`
- Phone: `916-409-2631`

Operational relevance:
- Use only when projected map location or downwind movement is plausibly toward the school area.
- If relevant, suggest notifying the school or relevant local authority.
- Do not claim the school is immediately threatened unless the map/distance/wind context supports it.

### Ferrari RCFE / senior care facility

Source status: observed on user screenshot and partially found via public web search, but not verified for runtime contact use.  
Operational relevance:
- Treat as a sensitive receptor only after verifying relevance to the projected incident area.
- Do not use unverified phone/email in live messages.

## Official emergency / local authority contacts

### Emergency

- For immediate danger to life/property: `911`.

### Lincoln Fire Department

- Non-emergency phone: `916-645-4040`.

Use as non-emergency fire contact only. For active emergency danger, recommend `911`.

### Lincoln Police Department

- Non-emergency phone: `916-645-4040`.
- The city page states Lincoln Police serves the City of Lincoln and that dispatch assists medical/law-enforcement requests mainly within city limits and also receives emergency calls from within Placer County.

Use as non-emergency local authority contact. For active emergency danger, recommend `911`.

### Placer County Office of Emergency Services

- Main phone: `530-886-5300`
- Emergency phone: `911`
- Toll-free: `800-488-4308 ext. 5300`

Use for county emergency-management context, preparedness, and coordination. Do not use it as the first substitute for `911`.

## Contact policy

The assistant must not invent real contacts.

If a contact is missing and operationally relevant, use one of these:
- "relevant local authority"
- "local emergency contact"
- "site operator"
- "property owner contact"

The assistant may ask:

> Do you want me to search for the relevant contact?

Perform web/API contact lookup only if the information is not available in the context data.

## Incident reasoning policy

The incident assistant should receive a structured context object before generating the visible first message.

Build context from:
1. detection label: `fire` or `smoke`
2. detection center / bounding box in image coordinates
3. image-to-map projection
4. zone membership and nearest zone
5. existing zone priorities from `live_ops_camera.json`
6. optional zone reference point from `live_ops_camera.json`
7. camera location and view direction
8. wind direction and downwind movement direction
9. landmarks and contact policy from this operational context

### First message style

The first message should be:
- short
- operational
- action-oriented
- not a full reasoning dump

Include only:
- what was detected
- where it is operationally
- likely drift / risk direction if known
- one recommended action question

Avoid by default:
- raw confidence
- raw pixel location
- raw coordinates
- temperature/humidity list
- full distance calculations
- detailed chain of reasoning

### Reasoning rules

- Inside or near a high-priority zone: raise urgency.
- Moving downwind toward residential, school, casino, farm, road, or senior-care area: raise urgency.
- Near wetland/water and moving away from structures: lower urgency but keep monitoring.
- Outside all zones but near a sensitive landmark: still treat as medium.
- Contact missing: use generic contact wording or ask whether to search.


## Example first messages

- `ThunderValleyWest detected smoke near Lincoln Crossing, drifting north. Do you want me to notify the relevant local authority?`
- `Smoke was detected near the farm owner site and appears to be moving toward Lincoln Crossing. Should I prepare a response update?`
- `Possible fire detected near the wetland area. It appears lower priority unless it spreads toward nearby structures. Should I keep monitoring?`
- `Smoke was detected outside the marked zones, but near a sensitive area and moving in that direction. Should I notify the relevant local authority?`

## Source links for manual verification

- Thunder Valley Casino Resort official site: https://www.thundervalleyresort.com/
- City of Lincoln Fire Department: https://www.lincolnca.gov/living-here/police-and-fire/fire-department/
- City of Lincoln Police Department: https://www.lincolnca.gov/living-here/police-and-fire/police-department/
- Placer County Office of Emergency Services: https://www.placer.ca.gov/1371/Office-of-Emergency-Services
- Lincoln Crossing Community Association: https://mylincolncrossing.org/page/16445~1054442
- Twelve Bridges High School: https://tbhs.wpusd.org/
