"""Context-aware reply composition.

The Hebbian network picks a reply *strategy* (Supportive, Upbeat, ...). This
module turns that strategy into words that fit what the person actually said:
it spots common life events ("I passed my test", "I got the job", "we broke
up") and writes a reply about that event in the chosen tone. Replies are
templates, not a language model.
"""
import re

# ---------------------------------------------------------------- events
# Each event: regex patterns, the emotion it usually carries, and replies per
# strategy. "{thing}" is filled with the matched noun (test, exam, job...).
_NOUN = r"(test|tests|exam|exams|paper|quiz|interview|course|class|semester|driving test|boards|finals|viva|project|assignment)"

EVENTS = [
    {
        "key": "passed",
        "emotion": "joy",
        "patterns": [
            rf"\b(passed|cleared|aced|cracked|topped|nailed)\b(?:\s+\w+){{0,3}}\s+{_NOUN}",
            rf"\b{_NOUN}\b.*\b(went (really )?well|results? (came|are) out.*(good|great|passed))",
            r"\b(got|scored)\b.*\b(good|great|full|top|high)\s+(marks|grades?|score|rank)\b",
            r"\bi passed\b",
        ],
        "replies": {
            "Upbeat": "Hurray! 🎉 You passed{the_thing} — you've done great! All that hard work really paid off. How are you going to celebrate?",
            "Curious": "You passed{the_thing}?! That's amazing! 🎉 Was it the result you were hoping for?",
            "Supportive": "You passed{the_thing} — well done! 🎉 You should be really proud of yourself.",
            "Neutral": "Congratulations on passing{the_thing}! 🎉 How do you feel now that it's done?",
        },
    },
    {
        "key": "new_job",
        "emotion": "joy",
        "patterns": [r"\b(got|landed|accepted|received)\b.*\b(job|offer|internship|placement|admission)\b",
                     r"\b(i'?m|i am|got)\s+(hired|selected|placed|accepted)\b"],
        "replies": {
            "Upbeat": "Woohoo! 🎉 Congratulations on the {thing}! You worked hard for this and it shows. When do you start?",
            "Curious": "No way — you got the {thing}?! 🎉 Tell me everything! How did you find out?",
            "Neutral": "Congratulations on the {thing}! 🎉 How are you feeling about it?",
        },
    },
    {
        "key": "promotion",
        "emotion": "joy",
        "patterns": [r"\b(got|been)\s+(a\s+)?promot(ed|ion)\b", r"\b(raise|hike|appraisal)\b.*\b(got|approved)\b"],
        "replies": {
            "Upbeat": "Yay, a promotion! 🎉 That's so well deserved — you've done great. How are you celebrating?",
            "Curious": "You got promoted?! 🎉 That's huge! What's the new role like?",
        },
    },
    {
        "key": "won",
        "emotion": "joy",
        "patterns": [r"\b(we|i)\s+won\b", r"\b(won|first place|first prize|gold medal|champion)\b"],
        "replies": {
            "Upbeat": "Yesss, you won! 🏆 That's brilliant — congratulations! How did it feel in that moment?",
            "Curious": "You won?! 🏆 Amazing! Tell me how it went!",
        },
    },
    {
        "key": "celebration",
        "emotion": "joy",
        "patterns": [r"\b(my|it'?s my)\s+birthday\b", r"\b(got|we'?re|i'?m)\s+(engaged|married)\b", r"\bgraduat(ed|ion)\b"],
        "replies": {
            "Upbeat": "Aww, that's wonderful! 🥳 Congratulations — I hope today feels as special as you are!",
            "Curious": "Oh, that's lovely! 🥳 What are your plans to celebrate?",
        },
    },
    {
        "key": "failed",
        "emotion": "sadness",
        "patterns": [rf"\b(failed|flunked|didn'?t pass|did not pass|bombed|messed up)\b(?:\s+\w+){{0,3}}\s+{_NOUN}",
                     r"\b(failed|flunked)\b", r"\b(low|bad|poor)\s+(marks|grades?|score)\b"],
        "replies": {
            "Supportive": "I'm really sorry{the_thing_didnt} 💗 One result doesn't define you or how capable you are. Do you want to talk about what happened?",
            "Calming": "That's a tough moment, and it's okay to feel upset. Take a breath — you can come back from this. What feels hardest right now?",
            "Neutral": "I'm sorry to hear that. How are you feeling about it?",
            "De-escalating": "It's completely understandable to be frustrated about that. Do you want to go over what felt unfair?",
        },
    },
    {
        "key": "rejected",
        "emotion": "sadness",
        "patterns": [r"\b(got\s+)?rejected\b", r"\b(didn'?t|did not)\s+get\s+(the\s+)?(job|offer|internship|admission|selected)\b"],
        "replies": {
            "Supportive": "I'm so sorry — rejection really stings 💗 It says nothing about your worth. Do you want to talk it through?",
            "Calming": "That's disappointing, and it's okay to feel it. Let's take it one step at a time — what would help most right now?",
        },
    },
    {
        "key": "scolded",
        "emotion": "sadness",
        "patterns": [r"\b(scolded|yelled at|shouted at|told off|blamed|insulted)\b"],
        "replies": {
            "Supportive": "That sounds really hurtful 💗 Nobody deserves to be spoken to like that. Do you want to tell me what happened?",
            "De-escalating": "That would make anyone upset. Let's slow down for a second — what part bothered you the most?",
            "Calming": "That sounds stressful. Take a slow breath with me — you're okay now. What happened?",
        },
    },
    {
        "key": "breakup",
        "emotion": "sadness",
        "patterns": [r"\b(broke up|break ?up|dumped|divorce|left me|cheated)\b"],
        "replies": {
            "Supportive": "I'm so sorry 💗 That's really painful. Be gentle with yourself — I'm here if you want to talk about it.",
            "Calming": "That's a lot to carry. It's okay to not be okay right now. What do you need most at the moment?",
        },
    },
    {
        "key": "loss",
        "emotion": "sadness",
        "patterns": [r"\b(passed away|died|lost my|funeral)\b"],
        "replies": {
            "Supportive": "I'm so, so sorry for your loss 💗 Please take all the time you need. Would you like to tell me about them?",
            "Calming": "I'm really sorry. Grief can be overwhelming — there's no right way to feel. I'm here with you.",
            "Neutral": "I'm very sorry for your loss. I'm here if you'd like to talk.",
        },
    },
    {
        "key": "sick",
        "emotion": "sadness",
        "patterns": [r"\b(i'?m|i am|feeling|feel)\s+(sick|ill|unwell)\b", r"\b(fever|headache|hospital)\b"],
        "replies": {
            "Supportive": "Oh no, I'm sorry you're not feeling well 💗 Please rest, drink some water, and see a doctor if it gets worse. Is anyone with you?",
            "Calming": "That sounds rough. Try to rest and take it easy today. Is there anything that would make you more comfortable?",
        },
    },
    {
        "key": "upcoming",
        "emotion": "fear",
        "patterns": [rf"\b{_NOUN}\b.*\b(tomorrow|next week|coming up|on monday|soon)\b",
                     rf"\b(nervous|scared|worried|anxious)\b.*\b{_NOUN}\b"],
        "replies": {
            "Calming": "It's completely normal to feel nervous before your {thing} 💗 Take a slow breath. You've prepared more than you think. Want to go over what's worrying you most?",
            "Supportive": "Feeling nervous just means it matters to you 💗 You've got this. Is there anything I can help you prepare?",
            "Upbeat": "You're going to do great in your {thing}! 💪 Want a quick plan to feel ready?",
        },
    },
    {
        "key": "argument",
        "emotion": "anger",
        "patterns": [r"\b(fight|fought|argument|argued)\b", r"\b(so unfair|not fair)\b"],
        "replies": {
            "De-escalating": "That sounds really frustrating, and your feelings are valid. Want to walk me through what happened so we can look at it calmly?",
            "Calming": "Arguments can leave you all wound up. Let's take a breath first — then tell me what happened.",
            "Supportive": "That sounds hard 💗 Fights with people we care about hurt. Do you want to talk about it?",
        },
    },
]


def detect_event(text):
    """Return (event, thing) for the first life event mentioned in text, else (None, None)."""
    cleaned = " ".join((text or "").lower().split())
    if not cleaned:
        return None, None
    for event in EVENTS:
        for pattern in event["patterns"]:
            match = re.search(pattern, cleaned)
            if match:
                noun = re.search(_NOUN, cleaned)
                thing = noun.group(0) if noun else ""
                if event["key"] in ("new_job",):
                    m = re.search(r"\b(job|offer|internship|placement|admission)\b", cleaned)
                    thing = m.group(0) if m else "job"
                return event, thing
    return None, None


# --------------------------------------------------------- generic replies
GENERIC = {
    "Supportive": [
        "That sounds really hard 💗 I'm here — do you want to talk it through, or would a small next step help more?",
        "I'm sorry you're going through this. You don't have to sort it all out at once. What's weighing on you most?",
    ],
    "De-escalating": [
        "I can hear that you're frustrated, and that's understandable. Want to walk me through what happened so we can look at it calmly?",
        "That sounds really annoying. Let's slow down for a second — what part bothers you most?",
    ],
    "Calming": [
        "It sounds like you're feeling anxious. Let's take it one piece at a time — try a slow breath, then tell me the first thing on your mind.",
        "That worry makes sense. What's the part you're most unsure about? We can break it down together.",
    ],
    "Upbeat": [
        "Yay, that's lovely to hear! 😊 What made it go so well?",
        "You sound really happy — I love that! ✨ Tell me more!",
    ],
    "Curious": [
        "Oh, that sounds unexpected! What happened?",
        "Wait, really? 😮 Tell me more!",
    ],
    "Neutral": [
        "I'm listening. What would be most useful for you right now?",
        "Thanks for sharing that with me. How are you feeling about it?",
    ],
}


def compose_reply(strategy, text="", variant=0, uncertainty=0.0):
    """Words for the chosen strategy, tailored to any event in the user's message."""
    if uncertainty > 0.8 and strategy == "Neutral" and not (text or "").strip():
        return "I couldn't quite catch that 🙈 Could you tell me in a few words how you're feeling?"
    event, thing = detect_event(text)
    if event and strategy in event["replies"]:
        the_thing = f" your {thing}" if thing else ""
        didnt = f" your {thing} didn't go the way you hoped" if thing else " it didn't go the way you hoped"
        return event["replies"][strategy].format(thing=thing or "it", the_thing=the_thing,
                                                 the_thing_didnt=didnt)
    options = GENERIC[strategy]
    return options[variant % len(options)]
