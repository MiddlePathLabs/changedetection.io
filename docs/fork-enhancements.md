# Additional monitoring features

## Reading and organising watches

History pages remember each watch's diff display settings across browsers and
devices. The AI "every change between versions" option resets on each visit.
Anonymous viewers of shared history links cannot change saved preferences.

Choose **Characters** alongside Lines and Words to highlight individual changed
digits or letters. The history-diff API also accepts `type=diffChars`.

The watch list remembers the selected group in the browser session. Select
**All** to clear it.

Select watches and choose **Download CSV** from the bulk actions to export their
current URLs, titles, errors, check times and price/stock metadata. See
[Watch-list CSV export](watch-list-export.md) for the report format.

## Stock phrases

In a restock watch's **Restock & Price Detection** tab, enter custom in-stock and
out-of-stock phrases, one literal phrase per line. Matching ignores case, accents
and repeated whitespace. A phrase must have word boundaries: "available" does
not match "unavailable". Script, style and metadata content are excluded.

A custom match overrides automatic availability detection while retaining price
data. If both kinds match, out-of-stock takes precedence. If neither matches,
automatic detection continues. Use specific phrases: matching a recommendation
or another product on the same page can give the wrong stock status. Group
restock overrides also support these fields.

## Failure and recovery alerts

Enable **Notify when checks repeatedly fail** in a watch's **Notifications** tab.
The threshold in **Settings** defaults to six consecutive failed checks. Set it
to zero to disable failure and recovery alerts globally.

One alert is sent when the threshold is reached, then further failures in the
same streak remain quiet. A successful check resets the streak, including a
successful check whose content is unchanged. **Notify when a failed watch
recovers** sends one recovery alert after a failure alert; it can be disabled
independently. Existing watches have failure alerts off until enabled.

Alerts use the watch's notification destinations, including applicable unmuted
notification groups. They can be sent before the first successful snapshot.
Existing filter and browser-step failure alerts remain separate; enabling both
can produce both kinds of alert for the same failure.

## Notification groups and proxies

A watch without its own notification destinations sends to every applicable
unmuted group with destinations. Each group's destination, format and templates
stay together. Watch-specific destinations retain priority.

Browser proxy credentials are percent-decoded before authentication. Proxy URLs
in the affected application debug logs have credentials redacted. This also
includes credential handling for the Scrapling stealth browser.
