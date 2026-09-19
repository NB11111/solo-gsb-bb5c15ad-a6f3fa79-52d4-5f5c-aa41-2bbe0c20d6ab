"""VALARM expansion for VEVENT instances.

Each :class:`~app.expander.Instance` produced by the recurrence expander
carries the raw VALARM components of the event that defines it:

* a master event's VALARMs apply to **every** expanded instance of the
  recurring series (master alarms are never re-attached per occurrence in
  the ICS text — one definition applies to the whole series);
* a ``RECURRENCE-ID`` override carries **its own** VALARMs and replaces the
  master's alarms for that instance (an override with no VALARM has no
  reminders).

Only ``ACTION:DISPLAY`` alarms are supported. ``TRIGGER`` may be absolute
(``VALUE=DATE-TIME``) or a positive/negative duration relative to DTSTART
(the default) or DTEND (``RELATED=END``). ``REPEAT``/``DURATION`` (which
must appear together per RFC 5545) fan the alarm out into repeated
reminders spaced by DURATION. Only reminders whose trigger instant falls
inside the half-open query window ``[window_start, window_end)`` are
returned.

A reminder can land inside the query window even when its event instance
does not overlap it: a ``TRIGGER:-P2D`` alarm fires two days *before* an
instance that may start after the window, and a positive or
``RELATED=END`` trigger can fire *after* an instance that already ended.
Each UID group is therefore expanded over a window widened by exactly the
reach of that group's relative triggers — the signed delta, the event
duration for ``RELATED=END`` anchors, and the ``REPEAT``/``DURATION``
fan-out — so every instance that can fire inside the window is
considered. Absolute ``DATE-TIME`` triggers are independent of the
instance time: they widen nothing and are still matched purely by trigger
time.

Anything invalid or unsupported raises :class:`AlarmError`, which the HTTP
layer maps to 422.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .expander import Instance
from .recurrence import (
    _event_times,
    _overlaps,
    _parse_calendar,
    _to_utc,
    compute_candidates,
    group_events_by_uid,
)

UTC = dt.timezone.utc

# Upper bound on reminders one alarm may fan out to (1 initial trigger plus
# REPEAT repeats), keeping an abusive REPEAT from flooding a response.
MAX_REMINDERS_PER_ALARM = 1_000


class AlarmError(Exception):
    """Invalid or unsupported VALARM input; surfaced to the client as 422."""


@dataclass(frozen=True)
class Reminder:
    """One concrete alarm trigger bound to a specific event instance."""

    uid: str
    recurrence_id: dt.datetime | None  # aware UTC; None for non-recurring events
    summary: str | None
    trigger: dt.datetime  # aware UTC
    description: str


@dataclass(frozen=True)
class _AlarmSpec:
    """A validated VALARM definition, independent of a concrete instance."""

    trigger: dt.datetime | dt.timedelta  # absolute datetime, or relative delta
    related_end: bool  # a relative trigger anchors to DTEND, not DTSTART
    description: str
    repeat: int  # additional repetitions after the first trigger
    spacing: dt.timedelta  # time between repeated triggers


def expand_alarms(
    ics_text: str, window_start: dt.datetime, window_end: dt.datetime
) -> list[Reminder]:
    """Expand every VALARM of every VEVENT instance in the UTC window.

    Instances come from the shared recurrence kernel
    (:func:`app.recurrence.compute_candidates`); an instance whose event
    has no VALARMs contributes nothing. All VALARMs in the calendar are
    validated eagerly (like the expander validates every event), so an
    unsupported alarm yields 422 even if its trigger would fall outside
    the window.

    Each UID group is expanded over the query window widened by that
    group's own alarm reach (see :func:`_window_padding`), so a reminder
    is still found when its trigger lands inside the window although the
    instance itself does not overlap it. Reminders are filtered by their
    trigger instant against the true (unpadded) window.
    """
    calendar = _parse_calendar(ics_text)

    specs: dict[int, _AlarmSpec] = {}
    for alarm in calendar.walk("VALARM"):
        specs[id(alarm)] = _parse_alarm(alarm)

    reminders: list[Reminder] = []
    for uid, events in group_events_by_uid(calendar).items():
        pad_before, pad_after = _window_padding(events, specs)
        padded_start = window_start - pad_before
        padded_end = window_end + pad_after
        # Same projection as app.expander.expand_parsed_calendar, over the
        # alarm-padded window: surviving instances overlapping it.
        for slot in compute_candidates(uid, events, padded_start, padded_end):
            instance = slot.instance
            if instance is None or not _overlaps(
                instance.start, instance.end, padded_start, padded_end
            ):
                continue
            for alarm in instance.alarms:
                spec = specs[id(alarm)]
                reminders.extend(
                    _materialize(spec, instance, window_start, window_end)
                )

    no_rid = dt.datetime.min.replace(tzinfo=UTC)
    reminders.sort(
        key=lambda r: (
            r.trigger,
            r.uid,
            r.recurrence_id or no_rid,
            r.description,
        )
    )
    return reminders


# ---------------------------------------------------------------------------
# Alarm-derived window padding
# ---------------------------------------------------------------------------


def _window_padding(
    events: list, specs: dict[int, _AlarmSpec]
) -> tuple[dt.timedelta, dt.timedelta]:
    """Expansion-window padding ``(before, after)`` for one UID group.

    Derived from the group's relative TRIGGERs — never a fixed number of
    days — so that every instance whose reminder could land inside the
    query window is expanded:

    * a duration trigger with delta ``d`` anchored at DTSTART fires
      between ``start + d`` and ``start + d + REPEAT * DURATION``; a
      ``RELATED=END`` anchor adds the event's duration to both bounds;
    * the latest bound reaching ``window_start`` dictates how far
      backwards instances must be expanded, and the earliest bound
      reaching ``window_end`` how far forwards.

    Absolute DATE-TIME triggers pin no instance time, so they contribute
    nothing (they are still filtered by trigger time). Durations follow
    the expander's rules: an override without DTEND/DURATION inherits the
    master event's duration.
    """
    masters = [e for e in events if "RECURRENCE-ID" not in e]
    master = masters[0] if masters else None
    master_duration: dt.timedelta | None = None
    if master is not None:
        start, end, _ = _event_times(master)
        master_duration = end - start

    before = after = dt.timedelta(0)
    for event in events:
        duration: dt.timedelta | None = None
        for component in event.subcomponents:
            if component.name != "VALARM":
                continue
            spec = specs[id(component)]
            if not isinstance(spec.trigger, dt.timedelta):
                continue  # absolute trigger: no instance-time reach
            if duration is None:
                if event is master or master is None:
                    start, end, _ = _event_times(event)
                else:
                    start, end, _ = _event_times(
                        event, default_duration=master_duration
                    )
                duration = end - start
            # Offsets of the first and last trigger from the instance
            # start (the DTEND anchor is folded in via the duration).
            first = spec.trigger + (duration if spec.related_end else dt.timedelta(0))
            last = first + spec.repeat * spec.spacing
            before = max(before, last)
            after = max(after, -first)
    return before, after


# ---------------------------------------------------------------------------
# VALARM parsing
# ---------------------------------------------------------------------------


def _parse_alarm(alarm) -> _AlarmSpec:
    action = alarm.get("ACTION")
    if action is None:
        raise AlarmError("VALARM is missing the required ACTION property")
    action = str(action).upper()
    if action != "DISPLAY":
        raise AlarmError(
            f"unsupported VALARM ACTION {action!r}; only DISPLAY is supported"
        )

    description = alarm.get("DESCRIPTION")
    if description is None:
        raise AlarmError("DISPLAY VALARM is missing the required DESCRIPTION")

    trigger_prop = alarm.get("TRIGGER")
    if trigger_prop is None:
        raise AlarmError("VALARM is missing the required TRIGGER property")
    params = trigger_prop.params or {}

    value_type = str(params.get("VALUE", "")).upper()
    raw = trigger_prop.dt
    related = str(params.get("RELATED", "")).upper()

    if isinstance(raw, dt.timedelta):
        if value_type not in ("", "DURATION"):
            raise AlarmError(
                f"TRIGGER VALUE={value_type} is incompatible with a duration value"
            )
        if related not in ("", "START", "END"):
            raise AlarmError(f"unsupported TRIGGER RELATED={related}")
        trigger: dt.datetime | dt.timedelta = raw
        related_end = related == "END"
    elif isinstance(raw, dt.datetime):
        if value_type not in ("", "DATE-TIME"):
            raise AlarmError(
                f"TRIGGER VALUE={value_type} is incompatible with a DATE-TIME value"
            )
        if related:
            raise AlarmError(
                "TRIGGER RELATED is only valid for duration-valued triggers"
            )
        trigger = _absolute_trigger(raw, params)
        related_end = False
    else:
        raise AlarmError(
            "unsupported TRIGGER value type: only DURATION and DATE-TIME are supported"
        )

    repeat, spacing = _parse_repeat(alarm)
    return _AlarmSpec(
        trigger=trigger,
        related_end=related_end,
        description=str(description),
        repeat=repeat,
        spacing=spacing,
    )


def _absolute_trigger(value: dt.datetime, params) -> dt.datetime:
    """Normalize an absolute TRIGGER value to an aware UTC datetime."""
    if value.tzinfo is None:
        tzid = params.get("TZID")
        if tzid:
            tzname = str(tzid)
            try:
                tz = ZoneInfo(tzname)
            except (ZoneInfoNotFoundError, ValueError, KeyError) as exc:
                raise AlarmError(f"unknown timezone: {tzname!r}") from exc
            value = value.replace(tzinfo=tz)
        else:
            # Floating time, like the expander's convention, is assumed UTC.
            value = value.replace(tzinfo=UTC)
    return _to_utc(value)


def _parse_repeat(alarm) -> tuple[int, dt.timedelta]:
    """Validate the REPEAT/DURATION pair; both or neither must be present."""
    repeat_prop = alarm.get("REPEAT")
    duration_prop = alarm.get("DURATION")
    if repeat_prop is None and duration_prop is None:
        return 0, dt.timedelta(0)
    if repeat_prop is None:
        raise AlarmError("VALARM DURATION requires REPEAT (they must appear together)")
    if duration_prop is None:
        raise AlarmError("VALARM REPEAT requires DURATION (they must appear together)")
    try:
        repeat = int(repeat_prop)
    except (TypeError, ValueError) as exc:
        raise AlarmError(f"invalid VALARM REPEAT value: {repeat_prop!r}") from exc
    if repeat < 0:
        raise AlarmError("VALARM REPEAT must not be negative")
    spacing = duration_prop.dt
    if not isinstance(spacing, dt.timedelta):
        raise AlarmError("VALARM DURATION must be a duration value")
    if spacing <= dt.timedelta(0):
        raise AlarmError("VALARM DURATION must be a positive duration")
    if repeat + 1 > MAX_REMINDERS_PER_ALARM:
        raise AlarmError(
            f"VALARM fans out to more than {MAX_REMINDERS_PER_ALARM} reminders"
        )
    return repeat, spacing


# ---------------------------------------------------------------------------
# Per-instance materialization
# ---------------------------------------------------------------------------


def _materialize(
    spec: _AlarmSpec,
    instance: Instance,
    window_start: dt.datetime,
    window_end: dt.datetime,
) -> list[Reminder]:
    if isinstance(spec.trigger, dt.datetime):
        first = spec.trigger
    else:
        anchor = instance.end if spec.related_end else instance.start
        first = anchor + spec.trigger

    reminders = []
    current = first
    for _ in range(spec.repeat + 1):
        if window_start <= current < window_end:
            reminders.append(
                Reminder(
                    uid=instance.uid,
                    recurrence_id=instance.recurrence_id,
                    summary=instance.summary,
                    trigger=current,
                    description=spec.description,
                )
            )
        current += spec.spacing
    return reminders
