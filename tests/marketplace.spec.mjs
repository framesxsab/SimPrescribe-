import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const baseUrl = process.env.SIMPLISCRIBE_BASE_URL || "http://127.0.0.1:8002";
const patientEmail = process.env.MARKETPLACE_PATIENT_EMAIL;
const patientPassword = process.env.MARKETPLACE_PATIENT_PASSWORD;
const pharmacyEmail = process.env.MARKETPLACE_PHARMACY_EMAIL;
const pharmacyPassword = process.env.MARKETPLACE_PHARMACY_PASSWORD;
const analysisId = process.env.MARKETPLACE_ANALYSIS_ID;
const pharmacyId = process.env.MARKETPLACE_PHARMACY_ID;

test("patient, pharmacist, fulfillment, and history lifecycle", async ({ browser }) => {
  test.setTimeout(120_000);
  const viewport = { width: 390, height: 844 };
  const patientContext = await browser.newContext({ viewport });
  const pharmacyContext = await browser.newContext({ viewport });
  const patientPage = await patientContext.newPage();
  const pharmacyPage = await pharmacyContext.newPage();

  async function login(page, email, password) {
    await page.goto(baseUrl + "/login", { waitUntil: "commit" });
    console.log("Marketplace E2E: login page committed");
    await expect(page.getByLabel("Email")).toBeVisible();
    console.log("Marketplace E2E: login form visible");
    await page.getByLabel("Email").fill(email);
    await page.getByLabel("Password").fill(password);
    await Promise.all([
      page.waitForURL((url) => (
        url.origin === new URL(baseUrl).origin && ["/", "/pharmacy"].includes(url.pathname)
      ), { waitUntil: "commit" }),
      page.getByRole("button", { name: "Sign in" }).click(),
    ]);
    console.log("Marketplace E2E: authenticated home visible");
  }

  async function expectAxeClean(page, state) {
    const results = await new AxeBuilder({ page }).analyze();
    const counts = Object.fromEntries(["critical", "serious", "moderate"].map(
      (impact) => [impact, results.violations.filter((violation) => violation.impact === impact).length],
    ));
    console.log(state + " Axe counts: " + JSON.stringify(counts));
    expect(results.violations, state + ": " + JSON.stringify(results.violations.map(
      ({ id, impact, nodes }) => ({ id, impact, targets: nodes.map((node) => node.target) }),
    ))).toEqual([]);
  }

  async function csrfPost(page, path, body) {
    return page.evaluate(async ({ requestPath, requestBody }) => {
      const source = [...document.scripts].map((script) => script.textContent || "").find((text) => text.includes("const csrf="));
      const token = source?.match(/const csrf='([^']+)'/)?.[1];
      if (!token) throw new Error("Could not read the page CSRF token.");
      const response = await fetch(requestPath, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": token },
        body: JSON.stringify(requestBody),
      });
      return { status: response.status, payload: await response.json() };
    }, { requestPath: path, requestBody: body });
  }

  try {
    console.log("Marketplace E2E: pharmacist login");
    await login(pharmacyPage, pharmacyEmail, pharmacyPassword);
    console.log("Marketplace E2E: pharmacist logged in");
    console.log("Marketplace E2E: source access before request");
    const beforeRequest = await pharmacyContext.request.get(baseUrl + "/api/analyses/" + analysisId + "/source");
    expect(beforeRequest.status()).toBe(403);

    console.log("Marketplace E2E: patient login");
    await login(patientPage, patientEmail, patientPassword);
    console.log("Marketplace E2E: patient logged in");
    await patientPage.setViewportSize(viewport);
    console.log("Marketplace E2E: open details");
    await patientPage.goto(baseUrl + "/details/" + analysisId, { waitUntil: "commit" });
    console.log("Marketplace E2E: details response committed");
    await expect(patientPage.getByRole("heading", { name: "Synthetic confirmed prescription.png" })).toBeVisible();
    await expect(patientPage.getByText("Ready to find a pharmacy")).toBeVisible();
    console.log("Marketplace E2E: details visible");
    await expectAxeClean(patientPage, "mobile confirmed prescription details");

    const findButton = patientPage.getByRole("button", { name: "Find pharmacies near me" });
    await findButton.click();
    console.log("Marketplace E2E: pharmacy search complete");
    const pharmacyCard = patientPage.locator("#pharmacy-results > div").filter({
      hasText: "Synthetic Marketplace Pharmacy",
    });
    await expect(pharmacyCard).toBeVisible();
    await expect(pharmacyCard.getByText("1/1 requested medicines currently listed")).toBeVisible();
    const orderButton = pharmacyCard.getByRole("button", { name: "Request pharmacist quote" });
    const navigation = patientPage.waitForURL(/\/orders\/[0-9a-f-]+$/);
    await orderButton.click();
    console.log("Marketplace E2E: order requested");
    await navigation;
    const orderUrl = patientPage.url();
    const orderId = orderUrl.split("/").at(-1);
    await expect(patientPage.getByRole("heading", { name: "Order " + orderId.slice(0, 8) })).toBeVisible();
    const patientOrderStatus = patientPage.locator("main > div:first-child span.rounded-full");
    await expect(patientOrderStatus).toHaveText("Requested");
    await patientPage.reload({ waitUntil: "commit" });
    await expect(patientOrderStatus).toHaveText("Requested");

    const pharmacySource = await pharmacyContext.request.get(baseUrl + "/api/analyses/" + analysisId + "/source");
    expect(pharmacySource.status()).toBe(200);
    expect(await pharmacySource.body()).toEqual(Buffer.from("synthetic-prescription-source"));

    await csrfPost(patientPage, "/api/orders", {
      analysis_id: analysisId,
      pharmacy_id: pharmacyId,
      fulfillment_mode: "pickup",
      delivery_address: "",
      generic_inquiries: [],
    }).then((response) => {
      expect(response.status).toBe(200);
      expect(response.payload).toEqual({ id: orderId, status: "requested", created: false });
    });

    console.log("Marketplace E2E: open pharmacy queue");
    await pharmacyPage.goto(baseUrl + "/pharmacy", { waitUntil: "commit" });
    await expect(pharmacyPage.getByRole("heading", { name: "Recent requests" })).toBeVisible();
    const orderLink = pharmacyPage.locator('a[href="/orders/' + orderId + '"]');
    await expect(orderLink).toBeVisible();
    await orderLink.click();
    await expect(pharmacyPage.getByRole("heading", { name: "Order " + orderId.slice(0, 8) })).toBeVisible();
    await expect(pharmacyPage.getByLabel("Exact inventory item")).toBeVisible();
    await expectAxeClean(pharmacyPage, "mobile pharmacist requested order");

    const itemRow = pharmacyPage.locator(".order-item").first();
    await itemRow.locator('[data-field="inventory_id"]').selectOption({ label: "Paracetamol 500 (3)" });
    await itemRow.locator('[data-field="verified_quantity"]').fill("2");
    await itemRow.locator('[data-field="price"]').fill("6.17");
    await itemRow.locator('[data-field="pharmacist_note"]').fill("Synthetic pharmacist verification");
    await pharmacyPage.getByRole("button", { name: "Submit verified quote" }).click();
    console.log("Marketplace E2E: quote submitted");
    const pharmacyOrderStatus = pharmacyPage.locator("main > div:first-child span.rounded-full");
    await expect(pharmacyOrderStatus).toHaveText("Quoted");
    await pharmacyPage.reload({ waitUntil: "commit" });
    await expect(pharmacyPage.getByText("2 × ₹6.17", { exact: false })).toBeVisible();

    const quotedResponse = await csrfPost(patientPage, "/api/orders", {
      analysis_id: analysisId,
      pharmacy_id: pharmacyId,
      fulfillment_mode: "pickup",
      delivery_address: "",
      generic_inquiries: [],
    });
    expect(quotedResponse.status).toBe(200);
    expect(quotedResponse.payload).toEqual({ id: orderId, status: "quoted", created: false });
    const quoteEventsResponse = await patientContext.request.get(baseUrl + "/api/orders/" + orderId);
    const quoteEvents = await quoteEventsResponse.json();
    expect(quoteEvents.events.filter((event) => event.status === "quoted")).toHaveLength(1);

    await patientPage.goto(orderUrl, { waitUntil: "commit" });
    await expect(patientPage.getByText("₹12.34", { exact: true })).toBeVisible();
    await expectAxeClean(patientPage, "mobile patient quote review");
    await patientPage.getByRole("button", { name: "Accept COD quote" }).click();
    await expect(patientOrderStatus).toHaveText("Accepted");
    const duplicateAcceptance = await csrfPost(patientPage, "/api/orders/" + orderId + "/accept", {});
    expect(duplicateAcceptance.status).toBe(409);

    await pharmacyPage.goto(orderUrl, { waitUntil: "commit" });
    await expect(pharmacyOrderStatus).toHaveText("Accepted");
    await pharmacyPage.getByRole("button", { name: "Start preparing" }).click();
    await expect(pharmacyOrderStatus).toHaveText("Preparing");
    await pharmacyPage.getByRole("button", { name: "Mark ready for pickup" }).click();
    await expect(pharmacyOrderStatus).toHaveText("Ready For Pickup");
    await pharmacyPage.getByRole("button", { name: "Mark fulfilled" }).click();
    await expect(pharmacyOrderStatus).toHaveText("Fulfilled");
    await pharmacyPage.goBack({ waitUntil: "commit" });
    await pharmacyPage.goForward({ waitUntil: "commit" });
    await pharmacyPage.reload({ waitUntil: "commit" });
    await expect(pharmacyOrderStatus).toHaveText("Fulfilled");
    await expectAxeClean(pharmacyPage, "mobile fulfilled pharmacy order");

    const inventoryResponse = await pharmacyContext.request.get(baseUrl + "/api/inventory");
    const inventory = await inventoryResponse.json();
    const savedItem = inventory.items.find((item) => item.medicine_name === "Paracetamol 500");
    expect(savedItem.stock_quantity).toBe(1);

    await patientPage.goto(baseUrl + "/orders", { waitUntil: "commit" });
    await expect(patientPage.getByRole("heading", { name: "Your orders" })).toBeVisible();
    await expect(patientPage.getByText("Synthetic Marketplace Pharmacy")).toBeVisible();
    await expect(patientPage.getByText("Paracetamol 500", { exact: true })).toBeVisible();
    await expect(patientPage.getByText("Fulfilled", { exact: true })).toBeVisible();
    await expectAxeClean(patientPage, "mobile patient order history");

    const finalOrder = await patientContext.request.get(baseUrl + "/api/orders/" + orderId);
    const finalData = await finalOrder.json();
    expect(finalData.status).toBe("fulfilled");
    expect(finalData.events.map((event) => event.status)).toEqual([
      "requested", "quoted", "accepted", "preparing", "ready_for_pickup", "fulfilled",
    ]);
    console.log("Marketplace lifecycle passed at 390px; inventory stock=1; timeline has six persisted events.");
  } finally {
    await patientContext.close();
    await pharmacyContext.close();
  }
});
