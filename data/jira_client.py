# -*- coding: utf-8 -*-
"""
Λεπτος client για το Jira Cloud REST API v3 — μονο stdlib (urllib).

ΣΗΜΕΙΩΣΗ ΣΥΜΒΑΤΟΤΗΤΑΣ: το παλιο /rest/api/3/search εχει ΑΦΑΙΡΕΘΕΙ απο το Jira Cloud
(HTTP 410). Χρησιμοποιειται το /rest/api/3/search/jql με pagination μεσω nextPageToken.

Το API v3 απαιτει ADF (Atlassian Document Format) για description/comment — οι helpers
adf() και adf_to_text() κανουν τη μετατροπη προς/απο απλο κειμενο.
"""
import base64
import json
import urllib.request
import urllib.error


# ── ADF helpers ───────────────────────────────────────────────────────
def adf(text: str) -> dict:
    """Απλο κειμενο -> ADF document (μια παραγραφος ανα γραμμη)."""
    paras = []
    for line in (text or "").split("\n"):
        if line.strip() == "":
            continue
        paras.append({"type": "paragraph",
                      "content": [{"type": "text", "text": line}]})
    if not paras:
        paras = [{"type": "paragraph", "content": []}]
    return {"type": "doc", "version": 1, "content": paras}


def adf_to_text(node) -> str:
    """ADF document -> απλο κειμενο."""
    if not isinstance(node, dict):
        return ""
    if node.get("type") == "text":
        return node.get("text", "")
    out = "".join(adf_to_text(c) for c in (node.get("content") or []))
    if node.get("type") in ("paragraph", "heading", "codeBlock",
                            "listItem", "blockquote"):
        out += "\n"
    return out


# ── Client ────────────────────────────────────────────────────────────
class JiraClient:
    def __init__(self, base_url: str, email: str, token: str):
        self.base = base_url.rstrip("/")
        cred = base64.b64encode(f"{email}:{token}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {cred}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _req(self, method: str, path: str, body=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base + path, data=data,
                                     headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:600]
            raise RuntimeError(f"Jira {method} {path} -> HTTP {e.code}: {detail}")

    # -- issues --------------------------------------------------------
    def create_issue(self, project_key, summary, description,
                     labels=None, issue_type="Task",
                     reporter_id=None, assignee_id=None):
        summary = " ".join((summary or "Untitled ticket").split())[:250]
        fields = {
            "project":   {"key": project_key},
            "summary":   summary,
            "description": adf(description),
            "issuetype": {"name": issue_type},
        }
        if labels:
            fields["labels"] = labels
        if reporter_id:
            fields["reporter"] = {"accountId": reporter_id}
        if assignee_id:
            fields["assignee"] = {"accountId": assignee_id}
        return self._req("POST", "/rest/api/3/issue", {"fields": fields})

    def search(self, jql, fields=None, page_size=50, max_total=500):
        """Search μεσω του ΝΕΟΥ endpoint, με pagination (nextPageToken)."""
        fields = fields or ["summary", "status", "labels", "description"]
        issues, token = [], None
        while len(issues) < max_total:
            body = {"jql": jql, "maxResults": page_size, "fields": fields}
            if token:
                body["nextPageToken"] = token
            d = self._req("POST", "/rest/api/3/search/jql", body)
            issues.extend(d.get("issues", []))
            token = d.get("nextPageToken")
            if not token:
                break
        return issues[:max_total]

    def add_comment(self, key, text, visibility_role=None):
        """Προσθετει σχολιο. Αν δοθει visibility_role, το σχολιο ειναι ορατο ΜΟΝΟ
        σε αυτον τον project role (π.χ. 'Administrators') — αλλιως δημοσιο."""
        body = {"body": adf(text)}
        if visibility_role:
            body["visibility"] = {"type": "role", "value": visibility_role}
        return self._req("POST", f"/rest/api/3/issue/{key}/comment", body)

    def add_labels(self, key, labels):
        return self._req("PUT", f"/rest/api/3/issue/{key}",
                         {"update": {"labels": [{"add": l} for l in labels]}})

    def assign(self, key, account_id):
        return self._req("PUT", f"/rest/api/3/issue/{key}/assignee",
                         {"accountId": account_id})

    # -- workflow ------------------------------------------------------
    def get_transitions(self, key):
        return self._req("GET", f"/rest/api/3/issue/{key}/transitions").get("transitions", [])

    def transition_by_name(self, key, candidate_names):
        """Δοκιμαζει ονοματα με τη σειρα· επιστρεφει αυτο που εφαρμοστηκε ή None."""
        available = self.get_transitions(key)
        by_name = {t["name"].strip().lower(): t["id"] for t in available}
        for name in candidate_names:
            tid = by_name.get(name.strip().lower())
            if tid:
                self._req("POST", f"/rest/api/3/issue/{key}/transitions",
                          {"transition": {"id": tid}})
                return name
        return None

    def myself(self):
        return self._req("GET", "/rest/api/3/myself")
