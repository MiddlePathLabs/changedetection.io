"""
AI setup assistant routes for the watch edit page - see llm/assist.py.

One POST per helper. Each is an explicit click, answers with JSON suggestions and never
changes the watch: the page puts accepted suggestions into the form, and the user saves.
"""

from flask import Blueprint, jsonify, request
from loguru import logger

from changedetectionio.auth_decorator import login_optionally_required
from changedetectionio.store import ChangeDetectionStore

ASSIST_KINDS = ('filters', 'noise', 'diagnose', 'tags')


def construct_blueprint(datastore: ChangeDetectionStore):
    ai_assist_blueprint = Blueprint('ui_ai_assist', __name__)

    @ai_assist_blueprint.route("/edit/<uuid_str:uuid>/ai-assist/<string:kind>", methods=['POST'])
    @login_optionally_required
    def watch_ai_assist(uuid, kind):
        from changedetectionio.llm import assist
        from changedetectionio.llm.evaluator import is_llm_features_disabled

        if is_llm_features_disabled():
            return jsonify({'status': 'error', 'error': 'AI features are disabled.'}), 404
        if kind not in ASSIST_KINDS:
            return jsonify({'status': 'error', 'error': 'Unknown AI helper.'}), 404
        watch = datastore.data['watching'].get(uuid)
        if not watch:
            return jsonify({'status': 'error', 'error': 'Watch not found.'}), 404

        try:
            if kind == 'filters':
                result = assist.suggest_filters(watch, datastore, goal=request.form.get('goal', '')[:2000])
            elif kind == 'noise':
                result = assist.suggest_noise(watch, datastore)
            elif kind == 'diagnose':
                result = assist.diagnose_error(watch, datastore)
            else:
                result = assist.suggest_tags(watch, datastore)
        except assist.AssistError as e:
            return jsonify({'status': 'error', 'error': e.message}), e.http_status
        except Exception as e:
            logger.exception(f"AI assist '{kind}' failed for {uuid}: {e}")
            return jsonify({'status': 'error', 'error': 'Unexpected error, see the application log.'}), 500

        if kind in ('filters', 'diagnose'):
            from changedetectionio.processors.text_json_diff import build_combined_filter_preview

            def values(field):
                return [item.get('value') for item in result.get(field, [])
                        if isinstance(item, dict) and item.get('value')]

            proposed = {}
            if kind == 'diagnose' and values('include_filters'):
                # Diagnosis calls these replacements, so preview the replacement
                # configuration rather than pretending an append is equivalent.
                proposed['include_filters'] = values('include_filters')
            else:
                for field in ('include_filters', 'subtractive_selectors', 'ignore_text', 'trigger_text'):
                    proposed[field] = list(dict.fromkeys(
                        list(watch.get(field) or []) + values(field)
                    ))
            result['combined_filter_preview'] = build_combined_filter_preview(
                datastore,
                watch,
                proposed_config=proposed,
                goal=request.form.get('goal', '')[:2000] or watch.get('llm_intent', ''),
            )

        return jsonify({'status': 'ok', 'kind': kind, **result})

    return ai_assist_blueprint
