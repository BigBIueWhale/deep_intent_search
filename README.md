# Deep Intent Search
Search algorithm that rivals the accuracy of a human reading through an article and highlighting relevant sections with a yellow marker

## Introduction

Existing solutions:

- 📏 Long context-length LLMs

- 🌲 Utilizing embedding LLMs to create a vector database.

- 🔢 Classic keyword-based lexical search algorithms

None of the aforementioned solutions are as thorough as manually reading a book or a series of articles chapter by chapter, and marking relevant information with a yellow marker 🖍️.

I can't afford missing **any** relevant information in the series of text documents I present, so I introduce `Deep Intent Search`.

For more information, see [search process](#search-through-the-chunks).

## Create .env

Create [.env](./.env) file containing:

```md
GOOGLE_AISTUDIO_API_KEY=XX123XX123_XXX123XXXXX123XXXXXXX123XXXX
CONTEXT_WINDOW_SIZE_TOKENS=8192
```

## Split the file

Use [semantic_splitter.py](./semantic_splitter.py) utility to create a folder at [./split](./split/).
A group of files with naming convention `[000001.txt, 000002.txt, ...]` will be created.

The splitting logic is done by utilizing structured output from `Gemini 2.5 Flash` in a recursive approach, to split the file into small chunks as the LLM sees fit.

```powershell
PS C:\Users\user\Downloads\deep_semantic_chunking> pip install -r requirements.txt
...
PS C:\Users\user\Downloads\deep_semantic_chunking> python semantic_splitter.py --file wikipedia_article.txt 
NLTK 'punkt' sentence tokenizer loaded.
--- Splitting file: wikipedia_article.txt ---
Original token count: 58256
Max tokens per chunk: 1024

Text token count (58256) exceeds window size (8192). Creating a central window for the LLM.
LLM identified a valid split point.
Text token count (23074) exceeds window size (8192). Creating a central window for the LLM.
LLM identified a valid split point.
Text token count (11967) exceeds window size (8192). Creating a central window for the LLM.
LLM identified a valid split point.
LLM identified a valid split point.
LLM identified a valid split point.
...
LLM identified a valid split point.
LLM identified a valid split point.
LLM identified a valid split point.
Text token count (11107) exceeds window size (8192). Creating a central window for the LLM.
LLM identified a valid split point.
LLM identified a valid split point.
...
LLM identified a valid split point.
LLM identified a valid split point.
LLM identified a valid split point.
...
LLM identified a valid split point.
LLM identified a valid split point.
LLM identified a valid split point.
LLM identified a valid split point.
LLM identified a valid split point.
LLM identified a valid split point.
LLM identified a valid split point.

--- Completed: Split into 104 chunks ---
Saving chunks to the 'split/' directory...

Saved chunk 1 to 'split\000001.txt' (Tokens: 584)
Saved chunk 2 to 'split\000002.txt' (Tokens: 368)
Saved chunk 3 to 'split\000003.txt' (Tokens: 719)
Saved chunk 4 to 'split\000004.txt' (Tokens: 582)
Saved chunk 5 to 'split\000005.txt' (Tokens: 435)
Saved chunk 6 to 'split\000006.txt' (Tokens: 312)
...
Saved chunk 15 to 'split\000015.txt' (Tokens: 646)
Saved chunk 16 to 'split\000016.txt' (Tokens: 374)
Saved chunk 17 to 'split\000017.txt' (Tokens: 530)
Saved chunk 18 to 'split\000018.txt' (Tokens: 543)
Saved chunk 19 to 'split\000019.txt' (Tokens: 861)
Saved chunk 20 to 'split\000020.txt' (Tokens: 645)
Saved chunk 21 to 'split\000021.txt' (Tokens: 689)
...
Saved chunk 93 to 'split\000093.txt' (Tokens: 642)
Saved chunk 94 to 'split\000094.txt' (Tokens: 600)
Saved chunk 95 to 'split\000095.txt' (Tokens: 291)
Saved chunk 96 to 'split\000096.txt' (Tokens: 800)
Saved chunk 97 to 'split\000097.txt' (Tokens: 691)
Saved chunk 98 to 'split\000098.txt' (Tokens: 420)
Saved chunk 99 to 'split\000099.txt' (Tokens: 2)
Saved chunk 100 to 'split\000100.txt' (Tokens: 125)
Saved chunk 101 to 'split\000101.txt' (Tokens: 569)
Saved chunk 102 to 'split\000102.txt' (Tokens: 699)
Saved chunk 103 to 'split\000103.txt' (Tokens: 474)
Saved chunk 104 to 'split\000104.txt' (Tokens: 306)

--- All chunks saved successfully. ---
PS C:\Users\user\Downloads\deep_search> 
```

## Search through the chunks
Once you have a folder [./split](./split/) containing ordered chunks of the larger file (or multiple concatenated files), it's time to perform the deep search.

We'll use [deep_search.py](./deep_search.py) utility to search based on intent, just like a human reading through a book.

The algorithm goes through each chunk (<1024 tokens) while providing up to `CONTEXT_WINDOW_SIZE_TOKENS` surrounding that chunk.

For example, if the current chunk is `000019.txt`, then the contents of surrounding adjacent chunks such as `[000016.txt, ..., 000019.txt, ..., 000022.txt]` will be included. This is important to give the LLM context regarding the meaning and significance of the (current) chunk of interest.

We want to avoid missing any relevant information, so the script makes a new and separate `Gemini 2.5 Flash` LLM completion request focusing on each chunk- meaning each chunk will be fed into the LLM multiple times in practice.

This is the most expensive possible way to search, and it's very unlikely to miss any relevant information about the search query.

Run this command:
```powershell
PS C:\Users\user\Downloads\deep_semantic_chunking> python deep_search.py --query "I already know about the Israel surprise attack on Iran, so I'm not interested in any generic information about that. But I am however interested in information that may be relevant to aircraft that may have been shot down by Iran (any Israeli aircraft) or other damages that may have been caused to Israeli military infrastructure that Israel might be hiding."
Loading and tokenizing all chunks from disk...
Successfully loaded 104 chunks into memory.

--- Starting Deep Search: Pass 1 (Context window: 8192 tokens) ---

Analyzing chunk 1/104 ('000001.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 2/104 ('000002.txt')...
  -> LLM decision: Not Relevant

...

Analyzing chunk 38/104 ('000038.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 39/104 ('000039.txt')...
  -> LLM decision: Relevant

Analyzing chunk 40/104 ('000040.txt')...
  -> LLM decision: Not Relevant

...

Analyzing chunk 66/104 ('000066.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 67/104 ('000067.txt')...
  -> LLM decision: Relevant

Analyzing chunk 68/104 ('000068.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 69/104 ('000069.txt')...
  -> LLM decision: Relevant

Analyzing chunk 70/104 ('000070.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 71/104 ('000071.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 72/104 ('000072.txt')...
  -> LLM decision: Relevant

Analyzing chunk 73/104 ('000073.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 74/104 ('000074.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 75/104 ('000075.txt')...
  -> LLM decision: Relevant

Analyzing chunk 76/104 ('000076.txt')...
  -> LLM decision: Not Relevant

...

Analyzing chunk 103/104 ('000103.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 104/104 ('000104.txt')...
  -> LLM decision: Not Relevant

================================================================================
Found 5 relevant section(s):
================================================================================

--- BEGIN: 000039.txt (Tokens: 402) ---
...
--- END: 000067.txt ---

--- BEGIN: 000069.txt (Tokens: 725) ---
...
--- END: 000069.txt ---

--- BEGIN: 000072.txt (Tokens: 815) ---
=== ישראל ===

על ישראל שוגרו למעלה מ-550 טילים ונשמעו בה למעלה מ-12,708 אזעקות. כ-86% מכלל הטילים ו-99% מהכטב"מים יורטו.{{הערה|שם=סיכום-ויינט}} כ-63 טילים הצליחו לחדור את [[מערך ההגנה האווירית|מערך ההגנה הרב-שכבתי]] וגרמו ל-29 הרוגים ו-3,238 פצועים, 23 מתוכם במצב קשה. 15,000 אזרחים פונו מבתיהם. נכון ל-24 ביוני 2025 הוגשו ל[[רשות המיסים]] 38,700 תביעות מהן 30,809 בגין נזק למבנה, 3,713 בגין נזק לרכב ו-4,085 בגין נזק לתכולה וציוד.{{הערה|שם=22-מיליארד|{{ynet|גד ליאור|12 יום שעלו 22 מיליארד שקל: המחיר הכלכלי של המלחמה|economy/article/yokra14417972|25 ביוני 2025}}}} עלות הנזקים הישירים הוערכה ביותר מ-4.5 [[מיליארד]] [[ש"ח]]. בנוסף נגרם נזק תשתיתי ונזק לבסיסים צבאיים, כולל פגיעות משמעותיות ב[[מכון ויצמן למדע]] בעלות של יותר משני מיליארד שקלים,<ref>{{וואלה|הודיה רן|"המעבדות נחרבו - ואנחנו נבנה מחדש": הנזק למכון ויצמן נאמד ביותר מ־2 מיליארד שקל|3759166|21 ביוני 2025}}</ref> במתחם [[בז"ן|בתי הזיקוק בחיפה]], ב[[בית החולים סורוקה]], ובמתקני תשתית חשמל ב[[אשקלון]] וב[[אשדוד]].{{הערה|שם=רון-בן-ישי-20-ביוני|{{ynet|רון בן ישי|הטריגר: מידע על כוונה לייצר 10,000 טילים עם פוטנציאל נזק של שתי פצצות אטום|news/article/yokra14412507|20 ביוני 2025}}}}<ref>{{ynet|ניצן גרידינגר|"מלחמת 12 הימים" - 29 הרוגים {{!}} המחיר הכבד ששילם העורף במערכה עם איראן|news/article/syvxux004gg|25 ביוני 2025}}</ref><ref>{{קישור כללי|כתובת=https://x.com/kann_news/status/1937585797810520322|כותרת=כאן חדשות C|אתר=X}}</ref> המחיר הכלכלי הכולל הוערך ב-22 מיליארד שקלים.{{הערה|שם=22-מיליארד}}
--- END: 000072.txt ---

--- BEGIN: 000075.txt (Tokens: 887) ---
'''כלי טיס של {{סמל|חיל האוויר הישראלי||+}}:'''
{| class="wikitable mw-collapsible mw-collapsed"
! תמונה
! שם
! סוג
!הערות
|-
| [[קובץ:IAF-F-15C-Baz--Independence-Day-2017-Tel-Nof-IZE-082.jpg|100 פיקסלים]]
| [[מקדונל דאגלס F-15 איגל|F-15 בז משופר]]
| [[מטוס קרב]] ל[[עליונות אווירית]]
|בשימוש הטייסות: [[טייסת 106|106]], [[טייסת 133|133]]
|-
| [[קובץ:רעם.jpg|100 פיקסלים]]
| [[F-15I רעם]]
| [[מטוס קרב רב-משימתי]] שיכול לשמש גם ל[[תקיפה אווירית]] וכן [[יירוט אווירי]]
|בשימוש טייסת [[טייסת 69|69]]
|-
| [[קובץ:IAF-F-16I-2016-12-13.jpg|100 פיקסלים]]
| [[F-16I סופה]]
| [[מטוס קרב רב-משימתי]] מדור 4.5
|בשימוש הטייסות: [[טייסת 107|107]], [[טייסת 119|119]], [[טייסת 201|201]], [[טייסת 253|253]],
|-
|[[קובץ:Operation Guardian of the Walls, May 2021 (51191995533).jpg|100 פיקסלים]]
|[[ג'נרל דיינמיקס F-16 פייטינג פלקון|F-16C/D]]
|מטוס קרב-רב משימתי מדור 4.5
|בשימוש הטייסות: [[טייסת 101|101]], [[טייסת 105|105]], [[טייסת 109|109]]<ref>{{קישור כללי|כתובת=https://www.iaf.org.il/9788-62416-he/IAF.aspx|הכותב=יותם רוזנפלד|כותרת=השמיים שלהם: הטייס והנווט שתקפו באיראן - וסגרו מעגל|אתר=אתר חיל האוויר|תאריך=20.06.2025}}</ref>    
|-
| [[קובץ:IAF-F-35I-Adir.jpg|100 פיקסלים]]
| [[F-35I אדיר]]
| [[מטוס קרב רב-משימתי]] [[חמקנות|חמקן]] מדור 5
|בשימוש הטייסות: [[טייסת 116|116]], [[טייסת 117|117]], [[טייסת 140|140]]
|-
| [[קובץ:IDF 122nd Squadron receiving the Oron aircraft, April 2021. I.jpg|100 פיקסלים]]
| [[גאלפסטרים 5|נחשון]]
| מטוסי [[לוחמה אלקטרונית]], [[מודיעין אותות]] ו[[בקרה אווירית]]
|בשימוש הטייסת [[טייסת 122|122]]
--- END: 000075.txt ---


================================================================================
You can now enter a new query to search within these results.
Press Enter to exit without a second pass.
================================================================================
Refinement Query > Only interested in specific sections that may contain information that might hint at damage to Israeli aircraft or military infrastructur
e. I'm not interested in a section that contains just general information about Israel's attack on Iran.

--- Starting Deep Search: Pass 2 (Refinement) ---

Analyzing chunk 1/5 ('000039.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 2/5 ('000067.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 3/5 ('000069.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 4/5 ('000072.txt')...
  -> LLM decision: Relevant

Analyzing chunk 5/5 ('000075.txt')...
  -> LLM decision: Not Relevant

================================================================================
Found 1 relevant section(s):
================================================================================

--- BEGIN: 000072.txt (Tokens: 815) ---
=== ישראל ===

על ישראל שוגרו למעלה מ-550 טילים ונשמעו בה למעלה מ-12,708 אזעקות. כ-86% מכלל הטילים ו-99% מהכטב"מים יורטו.{{הערה|שם=סיכום-ויינט}} כ-63 טילים הצליחו לחדור את [[מערך ההגנה האווירית|מערך ההגנה הרב-שכבתי]] וגרמו ל-29 הרוגים ו-3,238 פצועים, 23 מתוכם במצב קשה. 15,000 אזרחים פונו מבתיהם. נכון ל-24 ביוני 2025 הוגשו ל[[רשות המיסים]] 38,700 תביעות מהן 30,809 בגין נזק למבנה, 3,713 בגין נזק לרכב ו-4,085 בגין נזק לתכולה וציוד.{{הערה|שם=22-מיליארד|{{ynet|גד ליאור|12 יום שעלו 22 מיליארד שקל: המחיר הכלכלי של המלחמה|economy/article/yokra14417972|25 ביוני 2025}}}} עלות הנזקים הישירים הוערכה ביותר מ-4.5 [[מיליארד]] [[ש"ח]]. בנוסף נגרם נזק תשתיתי ונזק לבסיסים צבאיים, כולל פגיעות משמעותיות ב[[מכון ויצמן למדע]] בעלות של יותר משני מיליארד שקלים,<ref>{{וואלה|הודיה רן|"המעבדות נחרבו - ואנחנו נבנה מחדש": הנזק למכון ויצמן נאמד ביותר מ־2 מיליארד שקל|3759166|21 ביוני 2025}}</ref> במתחם [[בז"ן|בתי הזיקוק בחיפה]], ב[[בית החולים סורוקה]], ובמתקני תשתית חשמל ב[[אשקלון]] וב[[אשדוד]].{{הערה|שם=רון-בן-ישי-20-ביוני|{{ynet|רון בן ישי|הטריגר: מידע על כוונה לייצר 10,000 טילים עם פוטנציאל נזק של שתי פצצות אטום|news/article/yokra14412507|20 ביוני 2025}}}}<ref>{{ynet|ניצן גרידינגר|"מלחמת 12 הימים" - 29 הרוגים {{!}} המחיר הכבד ששילם העורף במערכה עם איראן|news/article/syvxux004gg|25 ביוני 2025}}</ref><ref>{{קישור כללי|כתובת=https://x.com/kann_news/status/1937585797810520322|כותרת=כאן חדשות C|אתר=X}}</ref> המחיר הכלכלי הכולל הוערך ב-22 מיליארד שקלים.{{הערה|שם=22-מיליארד}}
--- END: 000072.txt ---


--- Deep Search Complete ---
PS C:\Users\user\Downloads\deep_semantic_chunking> 
```
