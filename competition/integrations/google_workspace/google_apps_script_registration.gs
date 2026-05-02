const REGISTRATION_WEBHOOK_URL = "https://arguable-harpist-false.ngrok-free.dev/register";
const REGISTRATION_WEBHOOK_BEARER_TOKEN = "4c376ba89b8b8ade3626cb84252f2e35e1c02b2b31c70b630c03345ce4917f57";

// Organizer-managed content placeholders.
const SUBMISSION_FORM_LINK = "https://forms.gle/TNZb3grb1fJqq14m9";
const DISCORD_COMMUNITY_LINK = "<SET_DISCORD_LINK>";
const CONTACT_HELP_CHANNEL = "devclub.hcmus@gmail.com";
const CUSTOM_EMAIL_CONTENT = "<SET_CUSTOM_EMAIL_CONTENT>";

function onFormSubmit(e) {
  const values = e.namedValues || {};

  const payload = {
    "Team Name": singleValue(values["Team Name"]),
    "Primary contact name": singleValue(values["Primary contact name"]),
    "Primary contact email": singleValue(values["Primary contact email"]),
    "Second contact name": singleValue(values["Second contact name"]),
    "Second contact email": singleValue(values["Second contact email"]),
    "Agreement to rules": singleValue(values["Agreement to rules"]),
  };

  const response = UrlFetchApp.fetch(REGISTRATION_WEBHOOK_URL, {
    method: "post",
    contentType: "application/json",
    headers: {
      Authorization: "Bearer " + REGISTRATION_WEBHOOK_BEARER_TOKEN,
    },
    payload: JSON.stringify(payload),
    muteHttpExceptions: true,
  });

  const statusCode = response.getResponseCode();
  const bodyText = response.getContentText() || "{}";
  const result = JSON.parse(bodyText);

  if (statusCode !== 200 || result.status !== "success") {
    Logger.log("Registration webhook failed: code=" + statusCode + " body=" + bodyText);
    return;
  }

  // Build onboarding email with server-issued canonical identity and token.
  const email = payload["Primary contact email"];
  const teamName = result.team_name;
  const canonicalTeamId = result.canonical_team_id;
  const submissionToken = result.submission_token;

  const subject = "GDGoC AI Challenge 2026 - Registration Approved";
  const lines = [
    "Hello " + teamName + ",",
    "",
    "Your registration is approved.",
    "",
    "Team name: " + teamName,
    "Canonical team ID: " + canonicalTeamId,
    "Reusable submission token: " + submissionToken,
    "",
    "Submission form: " + SUBMISSION_FORM_LINK,
    "Discord/community link: " + DISCORD_COMMUNITY_LINK,
    "Contact/help channel: " + CONTACT_HELP_CHANNEL,
    "",
    "Submission constraints and format:",
    "- Upload exactly one .zip file.",
    "- The zip must contain exactly one agent.py.",
    "- No path traversal, no symlinks, no nested archives.",
    "",
    CUSTOM_EMAIL_CONTENT,
  ];

  MailApp.sendEmail({
    to: email,
    subject: subject,
    body: lines.join("\n"),
  });
}

function installOnSubmitTrigger() {
  const form = FormApp.getActiveForm();
  ScriptApp.newTrigger("onFormSubmit")
    .forForm(form)
    .onFormSubmit()
    .create();
}

function singleValue(valueArray) {
  if (!valueArray || valueArray.length === 0) {
    return "";
  }
  return String(valueArray[0]).trim();
}
