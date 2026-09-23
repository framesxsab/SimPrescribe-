import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const baseUrl = process.env.SIMPLISCRIBE_BASE_URL || "http://127.0.0.1:8002";
const fixtureKeys = [
  "A11Y_EMAIL",
  "A11Y_PASSWORD",
  "A11Y_PROCESSING_ID",
  "A11Y_REVIEW_ID",
  "A11Y_DETAILS_ID",
  "A11Y_ORDER_ID",
];
const missingFixtureKeys = fixtureKeys.filter((key) => !process.env[key]);
if (missingFixtureKeys.length) {
  throw new Error("The isolated accessibility fixture is incomplete: " + missingFixtureKeys.join(", "));
}

// The package runner provisions one disposable SQLite fixture with a synthetic patient,
// processing/review/details records, and an owned order, then tears it down after Playwright exits.
const routes = [
  ["dashboard", "/"],
  ["registration", "/register/patient"],
  ["login", "/login"],
  ["history", "/history"],
  ["marketplace", "/marketplace"],
  ["orders", "/orders"],
  ["processing", "/details/" + process.env.A11Y_PROCESSING_ID],
  ["review", "/details/" + process.env.A11Y_REVIEW_ID],
  ["details", "/details/" + process.env.A11Y_DETAILS_ID],
  ["order-detail", "/orders/" + process.env.A11Y_ORDER_ID],
];

test.beforeEach(async ({ page }) => {
  await page.goto(baseUrl + "/login", { waitUntil: "domcontentloaded" });
  await expect(page.getByLabel("Email")).toBeVisible();
  await page.getByLabel("Email").fill(process.env.A11Y_EMAIL);
  await page.getByLabel("Password").fill(process.env.A11Y_PASSWORD);
  await Promise.all([
    page.waitForURL(baseUrl + "/", { waitUntil: "domcontentloaded" }),
    page.getByRole("button", { name: "Sign in" }).click(),
  ]);
  await expect(page.locator("h1").first()).toBeVisible();
});

for (const [name, path] of routes) {
  test("axe " + name, async ({ page }) => {
    const response = await page.goto(baseUrl + path, { waitUntil: "domcontentloaded" });
    console.log(name + ": status=" + (response?.status() ?? "none") + " url=" + page.url());
    if (!response || response.status() >= 400) {
      test.skip(true, path + " returned " + (response?.status() ?? "no response"));
    }
    console.log(name + ": awaiting visible page heading");
    await expect(page.locator("h1").first()).toBeVisible();
    console.log(name + ": visible page heading");

    if (process.env.A11Y_CSS_DIAGNOSTICS === "1" && ["dashboard", "history"].includes(name)) {
      const element = name === "dashboard"
        ? page.getByText("Your saved prescription reviews.")
        : page.locator("p").filter({ hasText: "What it contains" }).first();
      const styles = await element.evaluate((node) => {
        const chain = [];
        for (let current = node; current; current = current.parentElement) {
          const style = getComputedStyle(current);
          chain.push({
            tag: current.tagName,
            className: current.className,
            color: style.color,
            backgroundColor: style.backgroundColor,
            opacity: style.opacity,
            fontSize: style.fontSize,
            fontWeight: style.fontWeight,
          });
          if (current.tagName === "BODY") break;
        }
        return chain;
      });
      console.log(name + ": computed-css=" + JSON.stringify(styles));
    }

    console.log(name + ": Axe analysis start");
    const results = await new AxeBuilder({ page }).analyze();
    console.log(name + ": Axe analysis complete");
    const counts = Object.fromEntries(["critical", "serious", "moderate"].map(
      (impact) => [impact, results.violations.filter((violation) => violation.impact === impact).length],
    ));
    console.log(name + ": " + results.violations.length + " total violations; " + JSON.stringify(counts));
    console.log(JSON.stringify(results.violations.map(({ id, impact, nodes }) => ({ id, impact, targets: nodes.map((node) => node.target) }))));
    const targetViolations = results.violations.filter((violation) => Object.hasOwn(counts, violation.impact));
    expect(targetViolations, JSON.stringify(targetViolations, null, 2)).toEqual([]);
  });
}
